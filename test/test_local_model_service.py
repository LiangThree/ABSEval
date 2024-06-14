import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import yaml
import ipdb

from llm_eval.proxy.services.local_model_service import LocalModelServcie


def test_single_model(model_name, path, request):
    print(f'testing model: {model_name}')
    service = LocalModelServcie(model_name, path)
    response = service.make_request(request)
    print(f'request result:\n {response}')
    if 'Chat' in model_name:
        response = service.chat(request)
        print(f'chat result:\n {response}')
    print('done')


def load_yaml_config():
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def test_local_model_service():
    request = '晚上睡不着怎么办？'
    config = load_yaml_config()
    
    for org, model_name_with_paths in config.items():
        if org != 'qwen': continue
        for model_name, paths in model_name_with_paths.items():
            path = paths[0]
            test_single_model('/'.join([org, model_name]), path, request)


if __name__ == "__main__":
    test_local_model_service()