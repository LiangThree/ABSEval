from vllm import LLM, SamplingParams
import torch
from typing import List
from llm_eval.data.request import RequestResult


class VllmModelChat:
    def __init__(self, model: LLM, model_name):
        self.model = model
        self.model_name = model_name

    def chat(self, prompts: List[str]) -> List[str]:
        pass


class BaichuanBaseChat(VllmModelChat):
    def __init__(self, model: LLM, model_name):
        super().__init__(model, model_name)

    def chat(self, prompts: List[str]) -> List[str]:
        sampling_params = SamplingParams(temperature=0.3, top_p=0.85, max_tokens=2048)
        reqs = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.build_chat_input(messages)
            reqs.append(input_ids)
            
        # generate completion_outputs
        completion_outputs = self.model.generate(reqs, sampling_params)
        
        # convert completion_outputs to responses
        # 这里VLLM的输出是CompletionOutput对象，需要转换成字符串
        responses = [completion_output.outputs[0].text for completion_output in completion_outputs]
        
        return responses

    def build_chat_input(self, messages: List[dict]):
        pass


class Baichuan13BChat(BaichuanBaseChat):
    def __init__(self, model: LLM, model_name):
        super().__init__(model, model_name)

    def chat(self, prompts: List[str]) -> List[str]:
        super().chat(prompts)

    def build_chat_input(self, messages: List[dict]):
        super().build_chat_input(messages)
        user_token_str = "<reserved_102>"
        assistant_token_str = "<reserved_103>"
        total_input = ""
        for message in reversed(messages):
            role_str = user_token_str if message['role'] == 'user' else assistant_token_str
            total_input = f"{role_str}{message['content']}{total_input}"

        # 在最后添加助手的标记，表示接下来的回复将属于助手
        total_input += assistant_token_str
        return total_input


class Baichuan2_13BChat(BaichuanBaseChat):
    def __init__(self, model: LLM, model_name):
        super().__init__(model, model_name)

    def chat(self, prompts: List[str]) -> List[str]:
        super().chat(prompts)

    def build_chat_input(self, messages: List[dict]):
        super().build_chat_input(messages)
        user_token_str = "<reserved_106>"
        assistant_token_str = "<reserved_107>"
        total_input = ""
        for message in reversed(messages):
            role_str = user_token_str if message['role'] == 'user' else assistant_token_str
            total_input = f"{role_str}{message['content']}{total_input}"

        # 在最后添加助手的标记，表示接下来的回复将属于助手
        total_input += assistant_token_str
        return total_input


class Baichuan2_7BChat(BaichuanBaseChat):
    def __init__(self, model: LLM, model_name):
        super().__init__(model, model_name)

    def chat(self, prompts: List[str]) -> List[str]:
        super().chat(prompts)

    def build_chat_input(self, messages: List[dict]):
        super().build_chat_input(messages)
        user_token_str = "<reserved_106>"
        assistant_token_str = "<reserved_107>"
        total_input = ""
        for message in reversed(messages):
            role_str = user_token_str if message['role'] == 'user' else assistant_token_str
            total_input = f"{role_str}{message['content']}{total_input}"

        # 在最后添加助手的标记，表示接下来的回复将属于助手
        total_input += assistant_token_str
        return total_input


_MODEL_REGISTRY = {
    "Baichuan-13B-Chat": Baichuan13BChat,
    "Baichuan2-7B-Chat": Baichuan2_7BChat,
    "Baichuan2-13B-Chat": Baichuan2_13BChat
}


def create_vllm_chat_model(model, model_name):
    if model_name in _MODEL_REGISTRY:
        chat_class = _MODEL_REGISTRY[model_name]
        return chat_class(model, model_name)
    else:
        raise ValueError(f"Model {model_name} is not supported.")


def check_model_in_support(model_name):
    return model_name in _MODEL_REGISTRY.keys()
