"""
本模块包含一个MultiChoiceJointAdapter类
该类继承自Adapter，用于将Instance对象转化为Request对象
其中最核心的操作是为instance构建合适的prompt
"""
import random
from typing import List

from llm_eval.data.instance import Instance
from llm_eval.data.request import Request, Prompt
from .adapter import Adapter

REFERENCE_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']


class MultiChoiceJointAdapter(Adapter):
    """
    Each `Instance` in a `Scenario` looks like this:

        <input> -> <reference1>
                   <reference2>
                   <reference3> [correct]
                   <reference4>

    We can define a label (e.g., letter) for each reference:

        <instructions>

        <input>                  # train_instance example_block
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        Answer: C

        <input>                  # test_instance prompt_block
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        Answer:
        
    In general, each example is:
    
        <input_prefix><input><reference_prefixes[0]><reference><output_prefix><output>
    """
    def __init__(self) -> None:
        super().__init__()
        self.name = 'multi_choice_joint_adapter'
        
    def adapt(self, instances: List[Instance]) -> List[Request]:
        train_instances = [i for i in instances if i.split == 'train']
        test_instances = [i for i in instances if i.split == 'test']
        few_shot_examples = self._sample_examples(train_instances)

        requests: List[Request] = []
        requests = [self._build_request(i, few_shot_examples) for i in test_instances]
        return requests

    def _build_request(self, instance: Instance, examples: List[Instance]):
        instruction_block = '请回答下面的选择题，你只需要给出正确选项，不要输出选项内容，也不要任何额外的解释：' + '\n'
        few_shot_block = self._build_few_shot_block(examples)
        prompt_block = self._build_prompt_block(instance)
        
        full_prompt_content = instruction_block + '\n' + few_shot_block + prompt_block
        prompt = Prompt(instance=instance, adapter_name=self.name, text=full_prompt_content)
        return Request(instance, prompt, question_type='multi_choice_question')

    @staticmethod
    def _build_prompt_block(instance: Instance):
        """
        根据instance构建prompt block
        <input>                  # test_instance prompt_block
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        Answer:
        """
        stem_block = instance.input.text + '\n'
        options_block = ''
        for label, reference in zip(REFERENCE_LABELS, instance.references):
            options_block += label + '. '
            options_block += reference.output.text
            options_block += '\n'
        options_block = options_block.strip() + '\n'
        prompt_block = '题目: ' + stem_block + options_block + '答案是: '
        return prompt_block

    def _build_few_shot_block(self, examples: List[Instance]):
        """
        根据examples构建few-shot block
        多个example_block之间用换行符分隔
        """
        few_shot_block = ''
        for example in examples:
            few_shot_block += self._build_example_block(example)
            few_shot_block += '\n'
        return few_shot_block

    @staticmethod
    def _build_example_block(example: Instance):
        """
        根据example和reference_index构建 example_block
        <input>                  # train_instance example_block
        A. <reference1>
        B. <reference2>
        C. <reference3>
        D. <reference4>
        Answer: C
        """
        stem_block = example.input.text + '\n'
        options_block = ''
        for label, reference in zip(REFERENCE_LABELS, example.references):
            options_block += label + '. '
            options_block += reference.output.text
            options_block += '\n'
        options_block = options_block.strip() + '\n'
        # 这里的answer部分应该是选项而不是文本，怎么做到的？
        # answer_block = example.correct_reference.output.text + '\n'
        answer_index = example.references.index(example.correct_reference)
        answer_block = REFERENCE_LABELS[answer_index] + '\n'
        return '题目: ' + stem_block + options_block + '答案是: ' + answer_block

    @staticmethod
    def _sample_examples(instances: List[Instance], k=5, seed=2023):
        """
        从instances里随机选择k个样例
        """
        random.seed(seed)
        return random.sample(instances, k=k)
