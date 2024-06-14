"""
该模块包含一个AdapterFactory类
该类用于根据method返回指定的Adapter对象
"""
from .adapter import Adapter
from .adapter_spec import AdapterSpec
from .generation_adapter import GenerationAdapter
from .multi_choice_joint_adapter import MultiChoiceJointAdapter


class AdapterFactory:
    @staticmethod
    def get_adapter(method: str) -> Adapter:
        if method == 'ADAPT_GENERATION':
            return GenerationAdapter()
        elif method == 'ADAPT_MULTI_CHOICE_JOINT':
            return MultiChoiceJointAdapter()
        raise ValueError(f'method {method} is not valid')
