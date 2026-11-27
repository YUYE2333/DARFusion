"""Epoch-boundary recovery without changing training randomness."""
from __future__ import annotations
import csv, hashlib, io, json, os, random, sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def environment_identity():
    return dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,cuda=torch.version.cuda)

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str,separators=(',',':')).encode()).hexdigest()

def training_code_hash():
    digest=hashlib.sha256()
    for folder in ('models','repair','engine','datasets','degradation','utils'):
        for path in sorted((ROOT/folder).glob('*.py')):
            digest.update(path.relative_to(ROOT).as_posix().encode());digest.update(path.read_bytes())
    digest.update((ROOT/'random_control.py').read_bytes())
    return digest.hexdigest()

def state_hash(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        t=tensor.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(t.dtype).encode());digest.update(str(tuple(t.shape)).encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def atomic_write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    try:
        with temporary.open('wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()

def atomic_json(path,value):
    atomic_write(path,(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n').encode('utf-8'))

def atomic_rows(path,rows):
    rows=list(rows)
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader();writer.writerows({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in row.items()} for row in rows)
    atomic_write(path,stream.getvalue().encode('utf-8-sig'))

def atomic_torch(path,value,*,rotate=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    try:
        with temporary.open('wb') as f:
            torch.save(value,f);f.flush();os.fsync(f.fileno())
        if rotate and path.is_file(): os.replace(path,path.with_name('resume_previous.pt'))
        os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()

def atomic_numpy(path,array):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    try:
        with temporary.open('wb') as f:
            np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()

def safe_load(path):
    return torch.load(path,map_location='cpu',weights_only=True)

def capture_random(loaders):
    ns=np.random.get_state()
    return {'python':random.getstate(),'numpy':[ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]],
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            'loaders':{k:l.generator.get_state() for k,l in loaders.items() if getattr(l,'generator',None) is not None}}

def restore_random(value,loaders):
    random.setstate(value['python']);ns=value['numpy']
    np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
    torch.set_rng_state(value['torch'])
    if value['cuda']:
        if not torch.cuda.is_available() or len(value['cuda'])!=torch.cuda.device_count():
            raise RuntimeError('CUDA device count changed; exact RNG recovery is unavailable')
        torch.cuda.set_rng_state_all(value['cuda'])
    for key,state in value['loaders'].items():
        generator=getattr(loaders[key],'generator',None)
        if generator is None: raise RuntimeError('DataLoader generator changed: '+key)
        generator.set_state(state)

class ResumeStore:
    def __init__(self,directory,identity,enabled=True):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'resume_latest.pt'
        self.identity=canonical_hash({'protocol':1,'identity':identity,'code':training_code_hash(),'environment':environment_identity()})
        self.enabled=enabled;self.preserve_previous=False
    def load(self):
        candidates=[p for p in (self.path,self.directory/'resume_previous.pt') if p.is_file()]
        if not candidates: return None
        if not self.enabled: raise RuntimeError('Recovery state already exists. Use --resume or a new output directory.')
        failures=[]
        for p in candidates:
            try: value=safe_load(p)
            except Exception as e:
                failures.append(str(p)+': '+str(e));continue
            if value.get('identity')!=self.identity:
                raise RuntimeError('Cannot resume: configuration, data, teacher or training code changed: '+str(p))
            if p!=self.path:
                self.preserve_previous=True
                print('[resume] latest unavailable; using previous complete epoch',flush=True)
            return value
        raise RuntimeError('No readable recovery checkpoint:\n'+'\n'.join(failures))
    def save(self,value):
        atomic_torch(self.path,{'identity':self.identity,**value},rotate=not self.preserve_previous)
        self.preserve_previous=False
        atomic_json(self.directory/'progress.json',{'epoch':value['epoch'],'goal_epochs':value['goal_epochs'],'identity':self.identity})

class EvaluationJournal:
    def __init__(self,path,identity,definitions,enabled=True):
        self.path=Path(path);self.identity=canonical_hash(identity);self.definitions=list(definitions);self.rows=[]
        if self.path.is_file():
            if not enabled: raise RuntimeError('Evaluation progress exists; use --resume')
            value=json.loads(self.path.read_text(encoding='utf-8'))
            if value['identity']!=self.identity: raise RuntimeError('Evaluation checkpoint/data/route changed')
            self.rows=value['rows']
            ids=[r['condition'] for r in self.rows]
            expected=[r['condition'] for r in self.definitions[:len(ids)]]
            if ids!=expected or len(set(ids))!=len(ids): raise RuntimeError('Invalid evaluation progress order')
    def add(self,row):
        if row['condition']!=self.definitions[len(self.rows)]['condition']: raise RuntimeError('Evaluation row out of order')
        self.rows.append(row);atomic_json(self.path,{'identity':self.identity,'rows':self.rows})

@contextmanager
def batch_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    f=path.open('a+b')
    if f.tell()==0: f.write(b' ');f.flush()
    f.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        f.close();raise RuntimeError('Another batch process is already using this output directory')
    try: yield
    finally:
        f.seek(0)
        if os.name=='nt':
            msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else: fcntl.flock(f.fileno(),fcntl.LOCK_UN)
        f.close()
