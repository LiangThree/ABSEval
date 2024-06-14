from typing import List, Union

import torch
import ipdb
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from .service import Service
from .chat_instruction_completor import ChatCompleterFactory, MetaChatCompleter
from .model_path import get_model_path
from .get_generation_config import get_generation_config
from .get_eos_token_id import get_eos_token_id

    
class LocalModelService(Service):
    def __init__(self, repo_id: str, model_config_path: str) -> None:
        self.repo_id = repo_id
        model_path = get_model_path(repo_id, model_config_path)
        print(f'loading model from {model_path}')
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
        )
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        except OSError:
            self.model.generation_config = GenerationConfig()
        print('model loading succeed')
        self.chat_completer: MetaChatCompleter = ChatCompleterFactory.create_chat_completer(repo_id, self.model, self.tokenizer)
        
    def make_request(self, prompt):
        """
        这里的generation_config是空的，用的是模型自定义配置
        """
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
                input_ids = inputs['input_ids'].to(self.model.device),
                # **generation_config.to_diff_dict(),
        )[0]
        response = outputs.tolist()[len(inputs["input_ids"][0]): ]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        return response_text
    
    def chat(self, prompt: Union[str, List[str]]):
        """
        chat方法用于处理单个prompt或者多个prompt的情况
        这里的prompt是一个字符串或者字符串列表
        """
        # 获取 chat 模式下的 generation config，如果没有chat模式的generation_config，就用默认生成模式下的generation_config
        # 生成stop_token_id
        generation_config: GenerationConfig = get_generation_config(self.repo_id, self.tokenizer)
        eos_token_id: List[int] = get_eos_token_id(self.repo_id, self.tokenizer)
        if eos_token_id is not None: generation_config.eos_token_id = eos_token_id
        # 这是正确的更新generation_config的方法，如果在generate方法传入generation_config
        # 遇到重复的参数就会报错
        self.model.generation_config.update(**generation_config.to_diff_dict())
        
        if isinstance(prompt, str):
            return self._one_chat(prompt)
        elif isinstance(prompt, list):
            return self._batch_chat(prompt)
        else:
            raise ValueError(f'prompt: {prompt} is not supported')
        
    def predict_multi_choice_question(self, prompt: str, num_options: int):
        model_org, model_name = self.repo_id.split('/')
        # get choice_ids by tokenizer
        choices= ["A", "B", "C", "D", 'E', 'F'][:num_options]
        if model_org == 'qwen':
            choice_ids = [self.tokenizer(choice)['input_ids'][0] for choice in choices]
        else:
            choice_ids = self.tokenizer.convert_tokens_to_ids(choices)
        
        # get last_token_logits using model forward method
        inputs = self.tokenizer(prompt, return_tensors='pt')
        # inputs = inputs['input_ids'].to(self.model.device)
        outputs = self.model(**inputs)
        last_token_logits = outputs.logits[:, -1, :]
        if model_org == 'qwen':
            last_token_logits = last_token_logits.to(dtype=torch.float32)
            
        # get choice_logits and pred
        choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
        pred = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}[np.argmax(choice_logits[0])]
        return pred
        
    def _batch_chat(self, prompts: List[str]) -> List[str]:
        """被chat方法调用，用于处理多个query的情况"""
        return [self._one_chat(prompt) for prompt in prompts]
        
    def _one_chat(self, prompt: str) -> str:
        """被chat方法调用，用于处理单个query的情况"""
        prompt = self.chat_completer.complete_chat_instruction(prompt, history=[])
        return self.make_request(prompt)


def main():
    model_path = '/home/zbl/data/llm/qwen/Qwen-7B-Chat'
    service = LocalModelService(model_path)
    res = service.make_request('世界上尚未发现的物种在未来十年内将会是什么，并且这些物种的发现将会对生物学产生何种影响？')
    print('=' * 100, '\n', res, '\n', '='*100)


if __name__ == '__main__':
    main()
