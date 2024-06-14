import torch
from transformers import GenerationConfig, PreTrainedTokenizer
from vllm import LLM, SamplingParams
from .chat_instruction_completor import ChatCompleterFactory, MetaChatCompleter
from .model_path import get_model_path
from typing import List, Union
from .get_generation_config import get_generation_config, get_sampling_params
from .get_eos_token_id import get_eos_token_id
import os


class VllmModelService:
    def __init__(self, repo_id: str, model_config_path: str, model_name: str) -> None:
        model_path = get_model_path(repo_id, model_config_path)
        self.repo_id = repo_id
        self.model_name = model_name
        print(f'loading model with vllm from {model_path}')
        self.vllm_model: LLM = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, gpu_memory_utilization=0.8)
        # self.vllm_model: LLM = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
        self.tokenizer: PreTrainedTokenizer = self.vllm_model.get_tokenizer()
        print('vllm model loading succeed')
        self.chat_completer: MetaChatCompleter = ChatCompleterFactory.create_chat_completer(repo_id, self.vllm_model, self.tokenizer)
        config_path = os.path.join(model_path, "generation_config.json")
        if os.path.exists(config_path):
            self.generation_config: GenerationConfig = GenerationConfig.from_pretrained(model_path)
        else:
            self.generation_config = GenerationConfig()  # 使用默认配置

    def make_request(self, prompts: Union[str, List[str]]):
        # sampling_params = SamplingParams(temperature=0.3, top_p=0.85, max_tokens=2048)
        sampling_params = get_sampling_params(self.repo_id, self.generation_config, self.tokenizer)
        eos_token_id: List[int] = get_eos_token_id(self.repo_id, self.tokenizer)
        if eos_token_id is not None: sampling_params.stop_token_ids = eos_token_id
        
        # convert prompts to input_ids
        if isinstance(prompts, str): prompts = [prompts]
        
        # generate completion_outputs
        completion_outputs = self.vllm_model.generate(prompts, sampling_params)
        
        # convert completion_outputs to responses
        # 这里VLLM的输出是CompletionOutput对象，需要转换成字符串
        responses = [completion_output.outputs[0].text.replace('json', '') for completion_output in completion_outputs]
        return responses

    def deal_str(self, answer):
        start = answer.find('{')  # 找到第一个左大括号的索引
        if start == -1:  # 如果找不到左大括号
            return answer
        end = answer.find('}', start)  # 从左大括号的索引开始找右大括号
        if end == -1:  # 如果找不到右大括号
            return answer

        return answer[start:end + 1]

    def chat(self, prompt: Union[str, List[str]]):
        """
        chat方法用于处理单个prompt或者多个prompt的情况
        这里的prompt是一个字符串或者字符串列表
        """
        if isinstance(prompt, str):
            prompt = self.chat_completer.complete_chat_instruction(prompt, history=[])
            prompt = [prompt]
        else:
            prompt = [self.chat_completer.complete_chat_instruction(p, history=[]) for p in prompt]

        response = self.make_request(prompt)
        response[0] = self.deal_str(response[0])
        return response
