import ipdb
from typing import List, Union
from .learner import Learner
from .statistic import Stat
from llm_eval.data.request import RequestResult, RequestMetric


class BasicLearner(Learner):
    def compute(
            self,
            request_results: Union[RequestResult, List[RequestResult]]
    ) -> RequestMetric:
        if isinstance(request_results, RequestResult):
            return self.compute_metric([request_results])
        elif isinstance(request_results, list):
            return [self.compute_metric(e) for e in request_results]
        else:
            raise ValueError(f'request_results type {type(request_results)} not valid')
    
    def compute_metric(self, request_result: RequestResult) -> RequestMetric:
        instance = request_result.request.instance
        correct_reference = instance.correct_reference
        completion = request_result.completion
        adapter_name = request_result.request.prompt.adapter_name
        correct_text = correct_reference.output.text
            
        stat = Stat(name='basic_metric')
        stat.add(correct_text == completion)
        eval = f'correct answer is: {correct_text}'
        return RequestMetric(request_result, stat, success=True, evaluation=eval)