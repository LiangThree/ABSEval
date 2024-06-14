"""
设置chat模式下的generation_config
"""

from transformers import GenerationConfig
from vllm import SamplingParams
def get_generation_config(repo_id, tokenizer=None):
    if repo_id == 'THUDM/chatglm3-6b':
        return get_chatglm3_generation_config()
    elif 'llama' in repo_id:
        return get_llama_generation_config(tokenizer)
    else:
        return set_default_generation_config()


def set_default_generation_config():
    kwargs = {
        'max_length': 8192,
    }
    return GenerationConfig(**kwargs)


def get_llama_generation_config(tokenizer):
    kwargs = {
        'pad_token_id': tokenizer.eos_token_id,
        'max_new_tokens': 8096
    }
    return GenerationConfig(**kwargs)


def get_chatglm3_generation_config():
    kwargs = {
        'max_length': 8192,
        'num_beams': 1,
        'do_sample': True,
        'top_p': 0.8,
        'temperature': 0.8,
        }
    return GenerationConfig(**kwargs)


def get_sampling_params(repo_id, generation_config, tokenizer=None):
    if repo_id == 'THUDM/chatglm3-6b':
        return get_chatglm3_sampling_params()
    elif 'llama' in repo_id:
        return get_llama_sampling_params(tokenizer)
    else:
        return set_default_sampling_params(generation_config)
        

def set_default_sampling_params(generation_config):
    config_dict = generation_config.to_dict()
    kwargs = {
        'max_tokens': 2048 if config_dict.get("max_length", 2048) < 512 else config_dict.get("max_length", 2048),
        'temperature': config_dict.get("temperature", 0.8),
        'top_p': config_dict.get("top_p", 0.8)
    }
    return SamplingParams(**kwargs)


def get_llama_sampling_params(tokenizer):
    # 下面的配置第一个可以防止llama输出`The attention mask and the pad token id were not set.` 的警告.
    # 第二个可以防止输出`Input length of input_ids is {}, but ``max_length`` is set to {}.`的警告
    kwargs = {
        'temperature': 0.8,
        'top_p': 0.8,
        'max_tokens': 8096
    }
    return SamplingParams(**kwargs)


def get_chatglm3_sampling_params():
    kwargs = {
        'max_tokens': 8192,
        'top_p': 0.8,
        'temperature': 0.8,
    }
    return SamplingParams(**kwargs)
