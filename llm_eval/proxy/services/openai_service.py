import time
import httpx
import ipdb
from dataclasses import dataclass

from openai import OpenAI, ChatCompletion
from func_timeout import func_timeout, FunctionTimedOut
from typing import Union, List
from tqdm import *

from .service import Service

api_key = ""


class OpenaiService(Service):
    def __init__(self, model_name):
        self.client = OpenAI(api_key=api_key)
        self.model_name = self.parse_model_name(model_name)
        
    def parse_model_name(self, model_name):
        if model_name in ['chatgpt', 'gpt-3.5-turbo']:
            return 'gpt-3.5-turbo'
        elif model_name == 'davinci-003':
            return 'text-davinci-003'
        elif model_name in 'gpt-4':
            return 'gpt-4'
        elif model_name in 'gpt-4-turbo':
            return 'gpt-4-turbo'
        elif model_name == 'gpt-3.5-turbo-16k':
            return 'gpt-3.5-turbo-16k'
        else:
            raise ValueError(f'model: {model_name} not found')
            
    def chat(self, prompt: Union[str, List[str]]):
        # print("openai chat")
        """
        chat方法用于处理单个prompt或者多个prompt的情况
        这里的prompt是一个字符串或者字符串列表
        """
        if isinstance(prompt, str):
            current_result = self.make_request(prompt)
            # print(current_result)
            return current_result
        elif isinstance(prompt, list):
            # print("here!")
            # exit(0)
            result = []
            for p in tqdm(prompt):
                current_result = self.make_request(p)
                # print(current_result)
                result.append(current_result)
            return result
        else:
            raise ValueError(f'prompt type: {type(prompt)} is not supported')

    def make_request(self, prompt: str, plain_text: bool = True):
        """
        每隔5秒请求一次，直到请求成功
        """
        while True:
            try:
                response: ChatCompletion = func_timeout(3000, self._send_request, args=(prompt,))
                return response

            except FunctionTimedOut:
                print('timeout, try again!')
            except Exception as e:
                print(f'Openai Error: {e}')
                print('sleep 5 second ...')
                time.sleep(5)

    def _send_request(self, prompt):
        """
        发送一次请求
        """
        if self.model_name in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4']:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                prompt=prompt
            )
        return response


if __name__ == '__main__':
    openai = OpenaiService('chatgpt')
    openai.chat('who are you?')
