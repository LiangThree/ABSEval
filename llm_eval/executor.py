"""
该模块实现了Executor类；
该类用于执行大模型的Inferece过程，也就是根据response输出模型的response；
该类是一个批量执行的过程，他的execute方法可以批量生成response并返回结果；
"""
import ipdb
from typing import List
from tqdm import tqdm
from pprint import pprint
from llm_eval.data.request import Request, RequestResult
from llm_eval.proxy.services.llm_service_factory import LLMServiceFactory


class Executor:
    def __init__(self, run_spec):
        model_repo_id = run_spec.model_conf['model_repo_id']
        accelerate_method = run_spec.scenario_conf.get("acceleration_method", "default")
        model_config_path = run_spec.model_conf.get('model_config_path', None)
        self.model = LLMServiceFactory.get_llm_service(model_repo_id, model_config_path, accelerate_method)
        self.run_spec = run_spec
        
    def execute(self, requests: List[Request]):
        request_results: List[RequestResult] = []
        qa_requests: List[Request] = []
        cq_requests: List[Request] = []
        
        # 这里为了进行qa推理加速，为避免调用错误，先将qa问题与选择问题分开
        for request in requests:
            if request.question_type == 'multi_choice_question':
                cq_requests.append(request)
            elif request.question_type == 'qa':
                qa_requests.append(request)
            else:
                raise ValueError(f'question_type: {request.question_type} not found')

        # 加速推理需全部request同时推理，此处判别是否qa并加速
        # 先处理选择问题
        if len(cq_requests) != 0:
            for cq_request in tqdm(cq_requests, desc="multi_choice_question executing..."):
                num_options = len(cq_request.instance.references)
                cq_response = self.model.predict_multi_choice_question(cq_request.prompt.text, num_options)
                # 将cq_response封装成RequestResult对象
                request_result: RequestResult = RequestResult(
                    request=cq_request,
                    success=True,
                    completion=cq_response
                )
                request_results.append(request_result)

        # 处理qa问题
        # qa问题的推理支持批量处理，因此需要将所有qa问题拼接成一个列表，然后一次性推理
        if len(qa_requests) != 0:
            # 为了加速，将所有qa问题拼接成一个列表，然后一次性推理
            with open('data/metrics/answer_question_prompt.txt', 'r') as file:
                prompt = file.read()
            # print(qa_requests[0].prompt)
            # print(qa_requests[0].prompt.instance)
            # print(qa_requests[0])
            # exit(0)
            prompts: List[str] = [prompt.replace('[MY_QUESTION]', qa_request.prompt.instance.references[0].output.text) for qa_request in qa_requests]
            pprint(prompts[:1])
            qa_responses: List[str] = self.model.chat(prompts)
            for index, qa_response in enumerate(qa_responses):
                index_of_one = qa_response.find('1')
                if index_of_one != -1:  # If '1' is found
                    qa_responses[index] = qa_response[index_of_one:]
                else:
                    pass  # If '1' is not found, return an empty string
            
            # 将cq_response封装成RequestResult对象
            for request, qa_response in zip(qa_requests, qa_responses):
                request_result = RequestResult(
                    request=request,
                    success=True,
                    completion=qa_response
                )
                request_results.append(request_result)
        return request_results
