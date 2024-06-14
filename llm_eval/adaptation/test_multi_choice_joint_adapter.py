import ipdb
from typing import List
from llm_eval.data.instance import Instance
from llm_eval.data.request import Request
from llm_eval.scenarios.wikidata_multi_choice_scenario import WikidataMultiChoiceScenario
from .multi_choice_joint_adapter import MultiChoiceJointAdapter


if __name__ == "__main__":
    scenario = WikidataMultiChoiceScenario()
    instances: List[Instance] = scenario.get_instances()
    multi_choice_joint_adapter = MultiChoiceJointAdapter()
    requests: List[Request] = multi_choice_joint_adapter.adapt(instances)
    ipdb.set_trace()
