import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import ipdb
import pprint
from typing import List
from llm_eval.scenarios import MultiChoiceScenario
from llm_eval.data.instance import Instance, Reference


def ensure_test_and_train_split(instances):
    test_num = 0
    for instance in instances:
        assert instance.split in ['test', 'train']
        if instance.split == 'test':
            test_num += 1
    assert test_num == 3
    
    
def ensure_references(instances: List[Instance]):
    for instance in instances:
        references = instance.references
        assert len(references) == 4
        for reference in references:
            assert isinstance(reference, Reference)
            

def ensure_correct_tag(instances: List[Instance]):
    for instance in instances:
        assert instance.correct_reference is not None
    

def test_multi_choice_scenario():
    db_path = 'data/knowledge.db'
    table_name = 'choices_question'
    target_view = 'knowledge-one-to-one-normal'
    scenario = MultiChoiceScenario(db_path, table_name, target_view)
    instances = scenario.get_instances()
    
    # 确保下列三个条件成立
    ensure_test_and_train_split(instances)
    ensure_references(instances)
    ensure_correct_tag(instances)
    
    # 查看样例
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(instances[0])


if __name__ == "__main__":
    test_multi_choice_scenario()