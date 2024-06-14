"""
已实现：baichuan, baichuan2, yi, qwen, llama
遇到bug：chatglm3, chatglm2
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ipdb
import json
from llm_eval.proxy.services.chat_instruction_completor import Baichuan2ChatCompleter, YiChatCompleter
from llm_eval.proxy.services.chat_instruction_completor import QwenChatCompleter, BaichuanChatCompleter
from llm_eval.proxy.services.chat_instruction_completor import LlamaChatCompleter, Chatglm3ChatCompleter
from transformers import LlamaTokenizer, AutoTokenizer
from llm_eval.proxy.services.llm_service_factory import LLMServiceFactory, get_model_path
from pathlib import Path
import importlib
from jinja2 import Template

from typing import List, Tuple, Dict

MODEL_CONFIG_PATH = "config/model_config.yaml"


def import_from_path(path: str, name: str='module'):
    """这里的name没什么用"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_instructions() -> list:
    """
    generate instructions for testing
    """
    instructions = []
    prompt = "你好，我是小明，你叫什么名字？"
    history = []
    instructions.append((prompt, history))
    
    prompt = "我是小明，你叫什么名字？"
    history = [
        {'role': 'user', 'content': '你好'}, 
        {'role': 'assistant', 'content': '你好，有什么可以帮助你的吗'}
    ]
    instructions.append((prompt, history))
    
    prompt = "你好，我是小明，你叫什么名字？"
    history = [
        {'role': 'system', 'content': '请扮演一个人工智能助手，并回答问题'}, 
        {'role': 'user', 'content': '你好'}, 
        {'role': 'assistant', 'content': '你好，有什么可以帮助你的吗'}
    ]
    instructions.append((prompt, history))
    
    return instructions


def test_baichuan_chat_completer(prompt: str, history: List[Dict[str, str]]):
    print(f'prompt: {prompt}, history: {history}')
    
    model_repo_id = "baichuan-inc/Baichuan-13B-Chat"
    
    # get tokenizer and model
    llm_service = LLMServiceFactory.get_llm_service(model_repo_id)
    model = llm_service.model
    tokenizer = llm_service.tokenizer
    
    def get_baichuan_gold_completion(prompt, history, model, tokenizer):
        # import build_chat_input from generation_utils.py
        messages = history.copy()
        messages.append({'role': 'user', 'content': prompt})
        if len(messages) > 0 and messages[0]['role'] == 'system':
            messages = messages[1:]
        gold_ids = model._build_chat_input(tokenizer, messages)
        gold_ids = gold_ids.tolist()[0]
        gold_text = tokenizer.decode(gold_ids)
        
        return gold_ids, gold_text
    
    # get gold completion
    gold_ids, gold_text = get_baichuan_gold_completion(prompt, history, model, tokenizer)
    print(f'gold_text: \n{gold_text}')
    
    # get generated completion
    chat_completer = BaichuanChatCompleter()
    generated = chat_completer.complete_chat_instruction(prompt, history)
    generated_ids = tokenizer.encode(generated)
    print(f'generated_text: \n{generated}')
    
    # # 这里tokenize的结果总是会多一个token：
    # ipdb> generated
    # '<reserved_102> 你好，我是小明，你叫什么名字？<reserved_103>'
    # ipdb> generated == gold_text
    # True
    # ipdb> tokenizer.encode('<reserved_102>')
    # [31106, 195]
    # ipdb> tokenizer.tokenize(generated_text)
    # *** NameError: name 'generated_text' is not defined
    # ipdb> tokenizer.tokenize(generated)
    # ['▁', '<reserved_102>', '▁你', '好', '，', '我是', '小', '明', '，', '你', '叫', '什么', '名字', '？', '<reserved_103>']
    
    # assert gold_ids == generated_ids, f'gold_ids: {gold_ids}, generated_ids: {generated_ids}'
    assert gold_text == generated, f'gold_text: {gold_text}, generated_text: {generated}'


def test_baichuan2_chat_completer(prompt: str, history: List[Dict[str, str]]):
    print(f'prompt: {prompt}, history: {history}')
    
    model_repo_id = "baichuan-inc/Baichuan2-7B-Chat"
    
    # get tokenizer and model
    llm_service = LLMServiceFactory.get_llm_service(model_repo_id)
    tokenizer = llm_service.tokenizer
    model = llm_service.model
    
    def get_baichuan_gold_completion(prompt, history, model, tokenizer):
        # import build_chat_input from generation_utils.py
        model_org, model_name = model_repo_id.split('/')
        model_path = get_model_path(model_org, model_name, MODEL_CONFIG_PATH)
        module_path = Path(model_path) / 'generation_utils.py'
        module = import_from_path(module_path)
        build_chat_input = module.build_chat_input
        
        # get gold completion
        messages = history.copy()
        messages.append({'role': 'user', 'content': prompt})
        gold_ids = build_chat_input(model, tokenizer, messages)
        gold_ids = list(gold_ids[0])
        gold_text = tokenizer.decode(gold_ids)
        
        return gold_ids, gold_text
    
    # get gold completion
    gold_ids, gold_text = get_baichuan_gold_completion(prompt, history, model, tokenizer)
    print(f'gold_text: \n{gold_text}')
    
    # get generated completion
    chat_completer = Baichuan2ChatCompleter()
    generated = chat_completer.complete_chat_instruction(prompt, history)
    generated_ids = tokenizer.encode(generated)
    print(f'generated_text: \n{generated}')
    
    assert gold_ids == generated_ids, f'gold_ids: {gold_ids}, generated_ids: {generated_ids}'
    assert gold_text == generated, f'gold_text: {gold_text}, generated_text: {generated}'
    
    
def test_yi_chat_completer(prompt: str, history: List[Dict[str, str]]):
    print(f'prompt: {prompt}, history: {history}')
    
    def get_yi_gold_completion(prompt, history):
        model_repo_id = "01ai/Yi-34B-Chat"
        model_org, model_name = model_repo_id.split('/')
        model_path = get_model_path(model_org, model_name, MODEL_CONFIG_PATH)
        tokenizer_config = json.load(open(os.path.join(model_path, 'tokenizer_config.json')))
        chat_template = tokenizer_config['chat_template']
        template = Template(chat_template)
        messages = history.copy()
        messages.append({'role': 'user', 'content': prompt})
        gold_text = template.render(messages=messages)
        return gold_text
    
    # get gold completion
    gold_text = get_yi_gold_completion(prompt, history)
    print(f'gold_text: \n{gold_text}')
    
    # get generated completion
    chat_completer = YiChatCompleter()
    generated = chat_completer.complete_chat_instruction(prompt, history)
    print(f'generated_text: \n{generated}')
    
    assert gold_text == generated, f'gold_text: {gold_text}, generated_text: {generated}'
    

def test_qwen_chat_completer(prompt: str, history: List[Dict[str, str]]):
    print(f'prompt: {prompt}, history: {history}')
    def build_qwen_history(history: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        """
        输入的格式为：[{'role': 'user', 'content': 'uuu'}, {'role': 'assistant', 'content': 'aaa'}]
        输出的格式为：[('uuu', 'aaa')]
        """
        result = []
        for item in history:
            if item['role'] == 'user':
                result.append((item['content'], ''))
            elif item['role'] == 'assistant':
                result[-1] = (result[-1][0], item['content'])
        return result
    
    def get_qwen_gold_completion(prompt, history):
        # get tokenizer and model
        model_repo_id = 'qwen/Qwen-7B-Chat'
        model_org, model_name = model_repo_id.split('/')
        model_path = get_model_path(model_org, model_name, MODEL_CONFIG_PATH)
        llm_service = LLMServiceFactory.get_llm_service(model_repo_id)
        tokenizer = llm_service.tokenizer
        
        # get gold completion
        module_path = os.path.join(model_path, 'qwen_generation_utils.py')
        module = import_from_path(module_path)
        make_context = module.make_context
        
        # get system text and history for qwen
        if len(history) > 0 and history[0]['role'] == 'system':
            system = history[0]['content']
            qwen_history = build_qwen_history(history[1:])
        else:
            system = "You are a helpful assistant."
            qwen_history = build_qwen_history(history)
        
        raw_text, context_tokens = make_context(tokenizer, prompt, qwen_history, system)
        return raw_text
                
        
    # get gold completion
    gold_text = get_qwen_gold_completion(prompt, history)
    print(f'gold_text: \n{gold_text}')
    
    # get generated completion
    chat_completer = QwenChatCompleter()
    history.insert(0, {'role': 'system', 'content': 'You are a helpful assistant.'})
    generated = chat_completer.complete_chat_instruction(prompt, history)
    print(f'generated_text: \n{generated}')
    
    assert gold_text == generated, f'gold_text: {gold_text}, generated_text: {generated}'
    
    
def test_llama_chat_completer(prompt: str, history: List[Dict[str, str]]):
    print(f'prompt: {prompt}, history: {history}')
    
    def get_llama_gold_completion(prompt, history):
        model_repo_id = "meta/llama2-7b-chat"
        model_org, model_name = model_repo_id.split('/')
        model_path = get_model_path(model_org, model_name, MODEL_CONFIG_PATH)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        chat_template = tokenizer.default_chat_template
        template = Template(chat_template)
        messages = history.copy()
        messages.append({'role': 'user', 'content': prompt})
        gold_text = template.render(
            messages=messages, 
            bos_token=tokenizer.bos_token,
            eos_token=tokenizer.eos_token
        )
        return gold_text
        
    # get gold completion
    gold_text = get_llama_gold_completion(prompt, history)
    print(f'gold_text: \n{gold_text}')
    
    # get generated completion
    chat_completer = LlamaChatCompleter()
    generated = chat_completer.complete_chat_instruction(prompt, history)
    print(f'generated_text: \n{generated}')
    
    assert gold_text == generated, f'gold_text: {gold_text}, generated_text: {generated}'
    
    
def test_chatglm3_chat_completer(prompt: str, history: List[Dict[str, str]]):
    print(f'prompt: {prompt}, history: {history}')
    
    def get_chatglm3_gold_completion(prompt, history):
        model_repo_id = "THUDM/chatglm3-6b"
        # llm_service = LLMServiceFactory.get_llm_service(model_repo_id)
        # tokenizer = llm_service.tokenizer
        model_org, model_name = model_repo_id.split('/')
        model_path = get_model_path(model_org, model_name, MODEL_CONFIG_PATH)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        gold_ids = tokenizer.build_chat_input(prompt, history)
        gold_ids = gold_ids['input_ids'].tolist()[0]
        gold_text = tokenizer.decode(gold_ids)
        return gold_text, gold_ids
        
    # get gold completion
    gold_text, gold_ids = get_chatglm3_gold_completion(prompt, history)
    print(f'gold_text: \n{gold_text}')
    
    # get generated completion
    chat_completer = Chatglm3ChatCompleter()
    generated = chat_completer.complete_chat_instruction(prompt, history)
    print(f'generated_text: \n{generated}')
    
    # 这里也有一个bug，有一个看起来像空格的字符，但是实际上不是空格
    assert gold_text == generated, f'gold_text: {gold_text}, generated_text: {generated}'
    
    
def test_chatglm2_chat_completer(prompt: str, history: List[Dict[str, str]]):
    print(f'prompt: {prompt}, history: {history}')
    
    def get_chatglm2_gold_completion(prompt, history):
        model_repo_id = "THUDM/chatglm2-6b"
        # llm_service = LLMServiceFactory.get_llm_service(model_repo_id)
        # tokenizer = llm_service.tokenizer
        model_org, model_name = model_repo_id.split('/')
        model_path = get_model_path(model_org, model_name, MODEL_CONFIG_PATH)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        gold_text = tokenizer.build_prompt(prompt, history)
        return gold_text
    
    # get gold completion
    gold_text = get_chatglm2_gold_completion(prompt, history)
    print(f'gold_text: \n{gold_text}')


def main():
    instructions: List[Tuple[str, List[Dict[str, str]]]] = create_instructions()
    for instruction in instructions:
        prompt, history= instruction
        test_chatglm3_chat_completer(prompt, history)

if __name__ == '__main__':
    main()