from __future__ import annotations
import copy, hashlib, json, os
from pathlib import Path
from repair.low_mix import settings
from utils.config import load_experiment

ROOT = Path(__file__).resolve().parents[1]


def selected_training(source):
    training = copy.deepcopy(source)
    model = training['robust_fusion']['model']
    if model.get('shared_expert_per_modality', False):
        raise ValueError('Use a configuration with independent family experts.')
    if model.get('hsi_degradation_encoder_variant', 'stats_only') != 'stats_only':
        raise ValueError('The selected model uses HSI stats_only.')
    if model.get('lidar_degradation_encoder_variant', 'statistics') != 'statistics':
        raise ValueError('The selected model uses LiDAR statistics.')
    robust = training['robust_fusion']['training']['robust_training']
    for key in ('feature_loss_weight', 'semantic_loss_weight', 'classification_loss_weight'):
        if float(robust.get(key, 1.0)) <= 0:
            raise ValueError(f'The selected model keeps {key}; use the Full configuration.')
    robust['composition'] = settings()
    return training


def relative_path(value):
    try: return os.path.relpath(Path(value).resolve(), ROOT)
    except ValueError: return str(Path(value).resolve())


def materialize(config, *, selection='official_test', data_root=None, split_file=None,
                model_seed=None, batch_size=None, device=None,
                stage1_epochs=None, stage2_epochs=None, validation_ratio=0.15):
    if not 0 < float(validation_ratio) < 1: raise ValueError('validation_ratio must be between 0 and 1')
    config = Path(config).expanduser().resolve()
    base = load_experiment(config, data_root_override=data_root, split_file_override=split_file)
    source = json.loads(config.read_text(encoding='utf-8'))
    training = selected_training(base['training'])
    if batch_size is not None: training['runtime']['batch_size'] = int(batch_size)
    if device is not None: training['runtime']['device'] = str(device)
    if stage1_epochs is not None: training['clean_fusion']['epochs'] = int(stage1_epochs)
    if stage2_epochs is not None: training['robust_fusion']['training']['robust_training']['epochs'] = int(stage2_epochs)
    experiment = copy.deepcopy(source)
    if model_seed is not None: experiment.setdefault('seeds', {})['model'] = int(model_seed)
    if selection == 'validation':
        experiment['validation'] = {**experiment.get('validation', {}), 'enabled':True, 'strategy':'stratified_random', 'ratio':float(validation_ratio)}
        experiment['selection_protocol'] = 'development_validation'
        training['robust_fusion']['training']['robust_training']['selection'] = 'validation_clean_plus_known_composite'
    elif selection == 'official_test':
        experiment['validation'] = {**experiment.get('validation', {}), 'enabled': False}
        experiment['selection_protocol'] = 'official_test_guided'
        training['robust_fusion']['training']['robust_training']['selection'] = 'official_test_clean_plus_known_composite'
    elif selection != 'config':
        raise ValueError('Unknown selection protocol: ' + str(selection))
    experiment['project'] = 'DARFusion'
    experiment['name'] = str(experiment.get('name', base['dataset']['dataset'])).replace('PCRFusion', 'DARFusion')
    dataset = copy.deepcopy(base['dataset'])
    dataset['default_root'] = relative_path(base['paths']['dataset_root'])
    dataset['label_split']['file'] = relative_path(base['paths']['label_split_file'])
    identity = dict(experiment=experiment,training=training,dataset=dataset,degradation=base['degradation'])
    token = hashlib.sha256(json.dumps(identity,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:16]
    folder = ROOT/'configs'/'model_generated'/token
    folder.mkdir(parents=True,exist_ok=True)
    experiment.update(dataset_config='dataset.json',degradation_config='degradation.json',training_config='training.json')
    for name,value in [('experiment.json',experiment),('training.json',training),('dataset.json',dataset),('degradation.json',base['degradation'])]:
        text = json.dumps(value,ensure_ascii=False,indent=2)+'\n'
        path = folder/name
        if path.is_file() and path.read_text(encoding='utf-8') != text:
            raise RuntimeError(f'Generated configuration conflict: {path}')
        path.write_text(text,encoding='utf-8')
    target = folder/'experiment.json'
    resolved = load_experiment(target)
    for key in ('dataset_root','label_split_file'):
        if resolved['paths'][key] != base['paths'][key]: raise AssertionError(f'Changed {key}')
    if resolved['config']['clean_fusion'] != training['clean_fusion']: raise AssertionError('Changed Stage I configuration')
    return target
