"""
本模块包含一个WikidataMultiChoiceScenario类
该类继承自Scenario类，实现了wikidata选择题类型的数据的读取
"""
import json
import random
from typing import List
from pathlib import Path
import ipdb

from .scenario import Scenario, TRAIN_SPLIT, TEST_SPLIT
from llm_eval.data.instance import Reference, CORRECT_TAG, Output
from llm_eval.data.instance import Instance, Input


class WikidataMultiChoiceScenario(Scenario):
    def get_instances(self) -> List[Instance]:
        """
        生成数据集的样本集合，每个样本Instance包含一个问题题干question和参考答案references；
        question是一个Input实例，包含问题的文本；
        references是一个Reference实例的列表，每个Reference实例包含一个参考答案的文本和tag；
        注意：训练集和测试集划分通过Instance的
        """
        root = Path('data/questions/2023-10-19-13-59/')
        file_paths = list(root.iterdir())

        questions: List[str] = []
        for file_path in file_paths:
            with open(file_path) as f:
                domain_questions = json.load(f)
            questions.extend(domain_questions)

        instances: List[Instance] = []
        for question in questions:
            references: List[Reference] = self._build_references(question)
            instance = Instance(
                input=Input(question['stem']),
                references=references
            )
            instances.append(instance)
        
        # 划分训练集和测试集
        instances = self._train_test_split(instances, train_ratio=0.2)
        
        return instances
    
    @staticmethod
    def _train_test_split(instances, train_ratio=0.2):
        random.shuffle(instances)
        num_train_samples = int(len(instances) * train_ratio)
        # 确保训练集至少有三个样本，用于后续mulchoice_joint_adapter的few-shot
        num_train_samples = max(num_train_samples, 5)
        for instance in instances[: num_train_samples]:
            instance.split = TRAIN_SPLIT
        for instance in instances[num_train_samples: ]:
            instance.split = TEST_SPLIT
        return instances

    @staticmethod
    def _build_references(question):
        """
        如果是生成式问题，生成问题的参考答案；
        如果是多选题，生成多问题的多个选项，其中正确选项的tag为CORRECT_TAG，其他选项的tag为None
        """
        options = ['A', 'B', 'C', 'D']
        references: List[Reference] = []
        for option in options:
            option_text = question[option]
            output = Output(option_text)
            reference = Reference(output)
            if option_text == question['answer']:
                reference.tag = CORRECT_TAG
            references.append(reference)
        return references


if __name__ == '__main__':
    scenario = WikidataMultiChoiceScenario()
    instances = scenario.get_instances()
    ipdb.set_trace()
