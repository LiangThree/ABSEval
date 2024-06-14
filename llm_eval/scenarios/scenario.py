"""
本模块包含一个Scenario类，该类是一个抽象类，定义了`get_instances`方法
注意Scenario类不应该涉及数据处理和问题生成的工作
数据处理和问题生成应该在data_process步骤完成，Scenario只涉及简单的数据读取和转换操作
"""
from abc import ABC, abstractmethod
from typing import List
from llm_eval.data.instance import Instance


# DATA SPLIT
TRAIN_SPLIT: str = "train"
TEST_SPLIT: str = "test"


class Scenario(ABC):
    @abstractmethod
    def get_instances(self) -> List[Instance]:
        pass
