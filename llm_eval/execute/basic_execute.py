import ipdb
from typing import List, Union
from .execute import Execute
from .statistic import Stat
from llm_eval.data.request import RequestResult, RequestMetric
from llm_eval.adaptation.multi_choice_joint_adapter import REFERENCE_LABELS


class BasicExecute(Execute):
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
        if adapter_name == 'multi_choice_joint_adapter':
            correct_index = instance.references.index(correct_reference)
            correct_text = REFERENCE_LABELS[correct_index]
        elif adapter_name == 'generation_adapter':
            correct_text = correct_reference.output.text
            
        stat = Stat(name='basic_metric')
        stat.add(correct_text == completion)
        eval = f'correct answer is: {correct_text}'
        return RequestMetric(request_result, stat, success=True, evaluation=eval)