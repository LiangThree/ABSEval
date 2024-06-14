"""设置chat模式下的stop_token_id
"""
def get_eos_token_id(repo_id: str, tokenizer):
    if 'qwen' in repo_id and '110' not in repo_id:
        return get_qwen_eos_token_id(tokenizer)
    elif repo_id == 'THUDM/chatglm3-6b':
        return get_chatglm3_eos_token_id(tokenizer)
    elif 'llama3' in repo_id.lower():
        return [128007, 128009]
    else:
        return None
    
    
def get_qwen_eos_token_id(tokenizer):
    return [tokenizer.im_end_id, tokenizer.im_start_id]


def get_chatglm3_eos_token_id(tokenizer):
    return [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command('<|observation|>')]