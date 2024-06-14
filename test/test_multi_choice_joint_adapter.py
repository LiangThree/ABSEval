import ipdb
from typing import List
from llm_eval.data.request import Request
from llm_eval.data.instance import Instance
from llm_eval.adaptation.multi_choice_joint_adapter import MultiChoiceJointAdapter
from llm_eval.scenarios.wikidata_multi_choice_scenario import WikidataMultiChoiceScenario


def test_multi_choice_joint_adapter(instances: List[Instance]):
    adapter = MultiChoiceJointAdapter()
    requests: List[Request] = adapter.adapt(instances)
    print(requests[0].prompt.text)
    
    
def main():
    scenario = WikidataMultiChoiceScenario()
    instances: List[Instance] = scenario.get_instances()
    test_multi_choice_joint_adapter(instances)
    
    
if __name__ == '__main__':
    main()