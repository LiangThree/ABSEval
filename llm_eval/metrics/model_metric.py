from pprint import pprint
from typing import List, Union
import ipdb
import json
import warnings
import sqlite3
from tqdm import *
from collections import OrderedDict

from .metric import Metric
from .statistic import Stat
from llm_eval.data.request import RequestResult, RequestMetric
from llm_eval.proxy.services.llm_service_factory import LLMServiceFactory
from llm_eval.proxy.services import RequestError
from llm_eval.proxy.services.service import Service
from data.database.util.database_util import EvalDataBase


class ModelMetric(Metric):
    def __init__(self, llm_service: Service, eval_prompt_format: str) -> None:
        # 加载评估模型
        self.model = llm_service

        # 解析 eval_prompt_format，从文件中读取，或者直接使用
        if eval_prompt_format.startswith('PATH:') or eval_prompt_format.startswith('path:'):
            path = eval_prompt_format[5:]
            with open(path, encoding='utf-8') as f:
                self.eval_prompt_format = f.read()
        else:
            self.eval_prompt_format = eval_prompt_format

    def compute(
            self,
            request_results: Union[RequestResult, List[RequestResult]],
            interfere=False
    ):
        if isinstance(request_results, RequestResult):
            self.compute_metrics([request_results], interfere)
        elif isinstance(request_results, list):
            self.compute_metrics(request_results, interfere)
        else:
            raise ValueError(f'request_results type {type(request_results)} not valid')

    def compute_metrics(self, request_results: List[RequestResult], interfere=False):

        eval_prompts: List[str] = [self.create_eval_prompt(request_result) for request_result in request_results]
        prompt_length = sum([len(prompt) for prompt in eval_prompts])

        if self.model.model_name == 'gpt-3.5-turbo':
            print(f"token_length:{prompt_length}     price:{round(prompt_length * 0.003 / 1000, 3)} $")
        elif self.model.model_name == 'gpt-4':
            print(f"token_length:{prompt_length}     price:{round(prompt_length * 0.03 / 1000, 3)} $")
        else:
            print(f"token_length:{prompt_length}     price:free")

        self.get_model_evaluation_n_times(request_results, eval_prompts, interfere)

    def create_stat_from_eval_response(self, eval_response: Union[str, None]) -> Stat:
        if eval_response is None:
            return self._create_error_stat()
        else:
            return self._create_success_stat(eval_response)

    def _create_success_stat(self, eval_response: str):
        stat = Stat(name='model_metric')
        eval_response_json = json.loads(eval_response)
        missing_steps = eval_response_json.get('missing_steps')
        if missing_steps == 'false':
            stat.add(1)
        else:
            stat.add(0)
        return stat

    def _create_error_stat(self):
        stat = Stat(name='model_metric')
        stat.add(0)
        return stat

    def get_model_evaluation_n_times(self, request_results: List[RequestResult], eval_prompts: List[str],
                                     interfere=False):
        # init eval_prompt_with_response with None
        eval_prompt_with_response = OrderedDict()

        for eval_prompt in eval_prompts:
            eval_prompt_with_response[eval_prompt] = None

        invalid_eval_count = 0
        todo_eval_prompts = [eval_prompt for eval_prompt, eval_response in eval_prompt_with_response.items() if
                             eval_response is None]

        eval_reponses = []
        request_lenth = len(request_results)

        if len(todo_eval_prompts) == 0:
            print('No inference to eval, next!')
            return

        from colorama import Fore
        print(Fore.BLUE + f'---------------------------------- config ----------------------------------')
        print(todo_eval_prompts[0])
        print(f'interfere:{interfere}')
        print(f'---------------------------------- end ----------------------------------' + Fore.RESET)

        for num in tqdm(range(request_lenth)):
            request_result = request_results[num]
            eval_prompt = todo_eval_prompts[num]

            question_id = request_result.request.instance.references[0].output.text[1]
            model_name = request_result.model_repo_id
            eval_model = self.model.model_name

            eval_response = self.model.chat(eval_prompt)
            if type(eval_response) == list:
                eval_response = eval_response[0]
            eval_reponses.append(eval_response)
            try:
                # self.ensure_anwer_valid(eval_response)
                eval_prompt_with_response[eval_prompt] = eval_response
                eval_result = json.loads(eval_response)
                missing_steps = eval_result['missing_steps']
                redundant_steps = eval_result['redundant_steps']
                duplicate_steps = eval_result['duplicate_steps']
                explain = eval_result['explain']

                if 'true' in missing_steps.lower():
                    missing_steps = 'True'
                else:
                    missing_steps = 'False'

                if 'true' in redundant_steps.lower():
                    redundant_steps = 'True'
                else:
                    redundant_steps = 'False'

                if 'true' in duplicate_steps.lower():
                    duplicate_steps = 'True'
                else:
                    duplicate_steps = 'False'

                data = [eval_model, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, explain]
                eval_db = EvalDataBase('data/database/script.db')
                if interfere:
                    # interfere删除eval_model_name元素
                    data.remove(data[0])
                    eval_db.update_interfere_eval_result(*data)
                else:
                    try:
                        eval_db.insert_into_eval_result(*data)
                    except sqlite3.IntegrityError:
                        pass

            except Exception as e:
                invalid_eval_count += 1
                print(e)
                print('invalid eval result:\n', eval_response)

        # 如果有不合法的eval_result，那么重新再来一次，否则跳出循环
        if invalid_eval_count > 0:
            print('get invalid eval result, invalid number:', invalid_eval_count)

        return eval_prompt_with_response.values()

    def create_eval_prompt(self, request_result: RequestResult) -> str:
        instance = request_result.request.instance
        completion = request_result.completion

        current_prompt = self.eval_prompt_format.replace('Question', instance.input.text)
        current_prompt = current_prompt.replace('Gold Answer', str(instance.references[0].output.text[2]))
        current_prompt = current_prompt.replace('Model Answer', str(completion))

        return current_prompt

    def ensure_anwer_valid(self, eval_response: str):
        """
        确保 eval_response 是一个json文件，并且有一个answer字段，该字段为0或1；
        """
        eval_response_json = json.loads(eval_response)

        missing_steps = eval_response_json.get('missing_steps').lower()
        redundant_steps = eval_response_json.get('redundant_steps').lower()
        duplicate_steps = eval_response_json.get('duplicate_steps').lower()
        explain = eval_response_json.get('explain')

        if missing_steps not in ['false', 'true']:
            raise ValueError(f'answer missing_steps:{missing_steps} not valid')
        if redundant_steps not in ['false', 'true']:
            raise ValueError(f'answer redundant_steps:{redundant_steps} not valid')
        if duplicate_steps not in ['false', 'true']:
            raise ValueError(f'answer duplicate_steps:{duplicate_steps} not valid')
