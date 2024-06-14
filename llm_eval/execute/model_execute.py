from pprint import pprint
from typing import List, Union
import ipdb
import json
import warnings
from collections import OrderedDict
from tqdm import *
import sqlite3
from .execute import Execute
from .statistic import Stat
from llm_eval.data.request import RequestResult, RequestMetric
from llm_eval.proxy.services.llm_service_factory import LLMServiceFactory
from llm_eval.proxy.services import RequestError
from llm_eval.proxy.services.service import Service
from data.database.util.database_util import EvalDataBase


class ModelExecute(Execute):
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
        # inference eval_prompts and get eval_responses
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
    
    def get_model_evaluation_n_times(self, request_results: List[RequestResult], eval_prompts: List[str], interfere=False):
        # init eval_prompt_with_response with None

        eval_prompt_with_response = OrderedDict()

        for eval_prompt in eval_prompts:
            eval_prompt_with_response[eval_prompt] = None

        invalid_eval_count = 0
        todo_eval_prompts = [eval_prompt for eval_prompt, eval_response in eval_prompt_with_response.items() if
                             eval_response is None]

        if len(todo_eval_prompts) == 0:
            print('No inference to eval, next!')
            return


        from colorama import Fore
        print(Fore.BLUE + f'---------------------------------- config ----------------------------------')
        print(todo_eval_prompts[0])
        print(f'interfere:{interfere}')
        print(f'---------------------------------- end ----------------------------------' + Fore.RESET)

        eval_responses = []
        request_length = len(request_results)
        for num in tqdm(range(request_length)):
            request_result = request_results[num]
            eval_prompt = todo_eval_prompts[num]

            eval_model = self.model.model_name
            question_id = request_result.request.instance.id
            model_name = request_result.model_repo_id
            eval_response = self.model.chat(eval_prompt)
            if type(eval_response) == list:
                eval_response = eval_response[0]
            eval_responses.append(eval_response)
            try:
                # self.ensure_anwer_valid(eval_response)
                eval_response.replace('TRUE', 'True')
                eval_response.replace('true', 'True')
                eval_response.replace('FALSE', 'False')
                eval_response.replace('false', 'False')
                eval_prompt_with_response[eval_prompt] = eval_response
                eval_result = json.loads(eval_response)
                limitation = eval_result['meet_limitation']
                complete = eval_result['complete_goal']
                step_order = eval_result['step_order_correct']
                explain = eval_result['explain']

                if 'true' in limitation.lower():
                    limitation = 'True'
                else:
                    limitation = 'False'

                if 'true' in complete.lower():
                    complete = 'True'
                else:
                    complete = 'False'

                if 'true' in step_order.lower():
                    step_order = 'True'
                else:
                    step_order = 'False'

                data = [eval_model, question_id, model_name, limitation, complete, step_order, explain]

                eval_db = EvalDataBase('data/database/script.db')
                if interfere:
                    # print(data)
                    eval_db.add_result_in_interfere(*data)

                else:
                    try:
                        eval_db.add_result_in_eval_result(*data)
                    except sqlite3.IntegrityError:
                        print('')
                        eval_db.update_eval_result(*data)

            except Exception as e:
                print(e)
                print(eval_response)
                invalid_eval_count += 1

        # 如果有不合法的eval_result，那么重新再来一次，否则跳出循环
        if invalid_eval_count > 0:
            print('get invalid eval result, invalid number', invalid_eval_count)

        return eval_prompt_with_response.values()
    
    def create_eval_prompt(self, request_result: RequestResult) -> str:
        instance = request_result.request.instance
        completion = request_result.completion

        current_prompt = self.eval_prompt_format.replace('QUESTION', instance.references[1].output.text)
        current_prompt = current_prompt.replace('LIMITATION', str(instance.references[0].output.text))
        current_prompt = current_prompt.replace('MODEL_INFERENCE', str(completion))


        return current_prompt
    
    def ensure_anwer_valid(self, eval_response: str):
        """
        确保 eval_response 是一个json文件，并且有一个answer字段，该字段为0或1；
        """
        eval_response_json = json.loads(eval_response)

        missing_steps = eval_response_json.get('meet_limitation').lower()
        redundant_steps = eval_response_json.get('complete_goal').lower()
        duplicate_steps = eval_response_json.get('step_order_correct').lower()
        explain = eval_response_json.get('explain').lower()

        if missing_steps not in ['false', 'true']:
            raise ValueError(f'answer limitation:{missing_steps} not valid')
        if redundant_steps not in ['false', 'true']:
            raise ValueError(f'answer complete:{redundant_steps} not valid')
        if duplicate_steps not in ['false', 'true']:
            raise ValueError(f'answer step_order:{duplicate_steps} not valid')

            
