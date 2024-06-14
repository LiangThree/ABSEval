from abc import ABC, abstractmethod
from typing import List, Union
from dataclasses import dataclass
from tqdm import tqdm

from llm_eval.data.request import RequestResult, RequestMetric


@dataclass
class ExecuteSpec:
    metric_name: str
    model_conf: dict = None
    eval_prompt_format: str = None


class Execute(ABC):
    def compute(
            self, 
            request_result: Union[RequestResult, List[RequestResult]]
    ) -> List[RequestMetric]:
        """
        request_result: inference结果
        request_metric: 评估结果
        """
        raise NotImplementedError