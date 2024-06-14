import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_eval.proxy.services import LLMServiceFactory
import argparse
import yaml
from typing import List

logging.basicConfig(level=logging.INFO, filename='test_llm_service_factory.log')

__ALL_MODELS__ = [
    "qwen/Qwen-72B-Chat",
    "meta/Llama-2-70b-chat",
    # "baichuan-inc/Baichuan2-13B-Chat", # ok
    # "01ai/Yi-34B-Chat", # ok
    # "tiiuae/falcon-40b-instruct" #ok
]
___INSTRUCTIONS__ = [
    # "晚上睡不着怎么办？",
    "1 + 2等于多少？",
    # "分析一下世界局势",
    # "who are you?"
]

def do_instructions(model):
    for prompt in ___INSTRUCTIONS__:
        print('-' * 20)
        print(f'prompt:{prompt}')
        response = model.chat(prompt)
        print(f'response: {response}')
    

def load_all_models(model_config_path: str) -> List[str]:
    """
    return: List[str]
    """
    with open(model_config_path, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    model_repo_ids = []
    for model_org, name_to_details in model_config.items():
        for model_name in name_to_details.keys():
            repo_id = f'{model_org}/{model_name}'
            model_repo_ids.append(repo_id)
    return model_repo_ids


def test_all_models(model_config_path: str = 'config/model_config.yaml'):
    model_repo_ids = __ALL_MODELS__
    for model_repo_id in model_repo_ids:
        print(f'testing {model_repo_id}')
        llm_service_factory = LLMServiceFactory()
        model = llm_service_factory.get_llm_service(
            model_repo_id=model_repo_id, 
            model_config_path=model_config_path,
            acceleration_method='vllm'
        )
        do_instructions(model)


def main():
    # get args
    args = argparse.ArgumentParser()
    args.add_argument("--chat", action="store_true")
    args.add_argument("--vllm", action="store_true")
    args.add_argument("--model_config_path", type=str, default='config/model_config_docker.yaml')
    args = args.parse_args()
    
    if args.vllm:
        args.acceleration_method = 'vllm'
    else:
        args.acceleration_method = 'default'
    
    test_all_models(args.model_config_path)

if __name__ == "__main__":
    logging.info('hello world')
    main()