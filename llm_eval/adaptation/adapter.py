"""
该模块包含一个Adapter类，
该类是一个抽象类，用于将Instance对象转化为Request对象
"""
from abc import ABC, abstractmethod
from typing import List

from llm_eval.data.instance import Instance
from llm_eval.data.request import Request


class Adapter:
    @abstractmethod
    def adapt(self, instance: List[Instance]) -> List[Request]:
        pass
