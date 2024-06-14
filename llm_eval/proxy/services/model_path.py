from pathlib import Path
import yaml


def get_model_path(repo_id, model_config_path='config/model_config.yaml'):
    """
    从model_config_path中加载文件路径，返回第一个存在的路径
    如果所有路径都不存在，则抛出异常
    """
    model_org, model_name = repo_id.split('/')
    with open(model_config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        
    try:
        path = config[model_org][model_name]['path']
    except KeyError:
        raise ValueError(f'model_org: {model_org} or model_name: {model_name} not found')
    
    if not Path(path).exists():
        raise ValueError(f'please ensure the path exists: {path}')
    return path