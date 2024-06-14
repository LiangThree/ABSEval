from typing import List, Union
import ipdb
import json
import warnings
from collections import OrderedDict
from tqdm import *

from colorama import Fore

from .learner import Learner, LearnerSpec
from .statistic import Stat
from llm_eval.data.request import RequestResult, RequestMetric, LearnerRequest
from llm_eval.proxy.services.llm_service_factory import LLMServiceFactory
from llm_eval.proxy.services import RequestError
from llm_eval.proxy.services.service import Service
from data.database.util.database_util import EvalDataBase


class ModelLearner(Learner):
    def __init__(self, model_conf, learner_spec: LearnerSpec, llm_service: Service, learner_prompt_format: str) -> None:
        # 加载评估模型
        self.model = llm_service
        self.learner_spec = learner_spec

        # 解析 eval_prompt_format，从文件中读取，或者直接使用
        if learner_prompt_format.startswith('PATH:') or learner_prompt_format.startswith('path:'):
            path = learner_prompt_format[5:]
            with open(path, encoding='utf-8') as f:
                self.learner_prompt_format = f.read()
        else:
            self.learner_prompt_format = learner_prompt_format


    def compute(
            self,
            request_results: Union[RequestResult, List[RequestResult]]
    ) -> List[LearnerRequest]:
        if isinstance(request_results, RequestResult):
            return self.compute_metrics([request_results])
        elif isinstance(request_results, list):
            return self.compute_metrics(request_results)
        else:
            raise ValueError(f'request_results type {type(request_results)} not valid')

    def compute_metrics(self, request_results: List[RequestResult]) -> List[RequestMetric]:
        # inference eval_prompts and get eval_responses
        eval_prompts: List[str] = [self.create_learner_prompt(request_result) for request_result in request_results]
        eval_ids: List[str] = [request_result.request.instance.id for request_result in request_results]


        from colorama import Fore
        print(Fore.BLUE + f'----------------------------------prompt example----------------------------------')
        if len(eval_prompts) > 5000:
            print(f"{eval_prompts[0][:5000]} \n ...")
        else:
            print(eval_prompts[0])
        print(f'----------------------------------end----------------------------------' + Fore.RESET)
        token_length = sum([len(prompt) for prompt in eval_prompts])
        if self.model.model_name == 'gpt-3.5-turbo':
            print(f"token_length:{token_length}     price:{round(token_length*0.003/1000,3)} $")
        elif self.model.model_name == 'gpt-4-turbo':
            print(f"token_length:{token_length}     price:{round(token_length*0.03/1000,3)} $")
        else:
            print(f"token_length:{token_length}     price:free")

        self.get_learn_result(eval_prompts, eval_ids)

    def get_learn_result(self, learn_prompts: List[str], eval_ids: List[str]):
        eval_db = EvalDataBase('data/database/script.db')
        # init eval_prompt_with_response with None
        learn_prompt_with_response = OrderedDict()
        for learn_prompt in learn_prompts:
            learn_prompt_with_response[learn_prompt] = None

        todo_learn_prompts = [eval_prompt for eval_prompt, eval_response in learn_prompt_with_response.items() if
                             eval_response is None]

        for i in tqdm(range(len(eval_ids))):
            prompt = todo_learn_prompts[i]
            question_id = eval_ids[i]

            eval_model_name = self.model.model_name

            # answer = eval_db.select_one_answer_from_gold_answer(question_id, eval_model_name)
            answer = eval_db.select_one_answer_from_gold_answer_without_learn(question_id, eval_model_name)
            if len(answer) == 0:
                learn_responses = self.model.chat(prompt)
                if type(learn_responses) == list:
                    learn_responses = learn_responses[0]
                # eval_db.insert_into_gold_answer(eval_model_name, question_id, learn_responses)
                eval_db.insert_into_gold_answer_without_learn(eval_model_name, question_id, learn_responses)
            else:
                print(f"{Fore.RED} gold answer exists, skip{Fore.RESET}")


    def create_learner_prompt(self, request_result: RequestResult) -> str:
        instance = request_result.request.instance
        completion = request_result.completion

        # 替换问题！
        prompt = self.learner_prompt_format.replace('Question', instance.references[0].output.text)
        # examples = ''
        # for index, answer in enumerate(completion[1]):
        #     examples += f'Example {index + 1}:\n'
        #     examples += f'{answer}\n\n'
        # prompt = prompt.replace('EXAMPLES', examples)

        return prompt



