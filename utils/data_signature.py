from __future__ import annotations
import hashlib,json,copy
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from datasets.common import MultimodalPatchDataset
from datasets.mat_hsi_lidar import MatArrayStore,align_label,_replace_label_values,_validate_single_split_maps,_validate_label_schema


def signature(dataset, spec):
    digest=hashlib.sha256()
    digest.update(np.ascontiguousarray(dataset.train_labels).tobytes())
    for key,value in sorted(dataset.norm_stats.items()):
        arr=np.ascontiguousarray(value)
        digest.update(key.encode()); digest.update(str(arr.shape).encode()); digest.update(str(arr.dtype).encode()); digest.update(arr.tobytes())
    files=[]
    for role,path in sorted(dataset.metadata.get('files',{}).items()):
        p=Path(path); st=p.stat()
        files.append((role,str(p.resolve()),st.st_size,st.st_mtime_ns))
    digest.update(json.dumps(dict(files=files,patch_size=spec.patch_size,classes=dataset.num_classes,
                                 bands=dataset.num_bands,lidar=dataset.lidar_channels,
                                 shape=list(dataset.hsi.shape)),sort_keys=True).encode())
    return digest.hexdigest()


def provider_cache(bundle,dataset,spec):
    return Path(bundle['paths']['project_root'])/'.cache'/'composition_degradations'/dataset.dataset_name/signature(dataset,spec)[:16]


def evaluation_loader(bundle,dataset,spec,configured_loader,split='official_test'):
    validation=bool(bundle['config']['validation'].get('enabled',False))
    if split=='validation':
        if not validation: raise ValueError('Validation evaluation requires validation.enabled=true')
        return configured_loader,'validation'
    if split=='configured': return configured_loader,dataset.metadata.get('split',{}).get('evaluation_split','configured_test')
    if not validation: return configured_loader,'official_test'
    # Load only official test labels, retaining train_dev normalization and PCA.
    cfg=bundle['dataset']; label_cfg=cfg['label_split']
    store=MatArrayStore(spec.data_root,allow_implicit_key_fallback=bool(cfg['resolution']['allow_implicit_key_fallback']))
    labels=align_label(store.load(label_cfg['file'],label_cfg['keys']['test'],'test_labels',2),
                       'test_labels',str(cfg['layouts']['labels']),tuple(dataset.hsi.shape[:2]))
    labels=_replace_label_values(labels,dict(cfg['label_values']['replace']))
    _validate_single_split_maps(dataset.train_labels,labels,None)
    _validate_single_split_maps(dataset.test_labels,labels,None)
    _validate_label_schema(dataset.train_labels,labels,dict(cfg['label_schema']))
    patches=MultimodalPatchDataset(dataset.hsi,dataset.lidar,labels,np.argwhere(labels>0),spec.patch_size)
    return DataLoader(patches,batch_size=spec.batch_size,shuffle=False,num_workers=spec.num_workers,
                      pin_memory=spec.pin_memory),'official_test'

def code_signature():
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for name in ('repair/low_mix.py','utils/data_signature.py','utils/final_config.py','models/pcrfusion.py',
                 'models/clean_fusion.py','engine/train_robust_fusion.py','engine/train_clean_fusion.py',
                 'datasets/mat_hsi_lidar.py','utils/config.py'):
        digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()
