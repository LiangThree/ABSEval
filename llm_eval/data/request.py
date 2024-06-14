from dataclasses import dataclass, field
from .instance import Instance
from llm_eval.metrics.statistic import Stat

from typing import Optional, Dict

@dataclass
class Prompt:
    instance: Instance
    adapter_name: str
    text: str

@dataclass
class Request:
    instance: Instance
    prompt: Prompt
    question_type: str

@dataclass
class RequestLearner:
    request: Request
    success: bool
    completion: str

@dataclass
class RequestResult:
    request: Request
    success: bool
    completion: str
    model_repo_id: str = None

@dataclass
class RequestMetric:
    request_result: RequestResult
    stat: Stat
    success: bool
    evaluation: str = ''

@dataclass
class LearnerRequest:
    request_result: RequestResult
    learn_response: str

def request_metric_to_dict(request_metric: RequestMetric) -> Dict:
    instance = request_metric.request_result.request.instance
    prompt = request_metric.request_result.request.prompt
    completion = request_metric.request_result.completion
    stat = request_metric.stat.mean
    return {'prompt': prompt.text, 'completion': completion, 'stat': stat}
