"""
该模块实现了ScenarioFactory类，用于根据scenario_name返回指定的Scenario对象
"""
from .wikidata_scenario import WikidataScenario
from .wikidata_multi_choice_scenario import WikidataMultiChoiceScenario
from .multi_choice_scenario import MultiChoiceScenario
from .question_answering_scenario import QuestionAnsweringScenario
from .scenario import Scenario


class ScenarioFactory:
    @staticmethod
    def get_scenario_from_dict(scenario_conf: dict) -> Scenario:
        # 从scenario_conf中获取scenario_name
        scenario_name = scenario_conf.pop('scenario_name')
        
        # 根据scenario_name，和scenario_conf中的其他参数返回指定的Scenario对象
        if scenario_name == 'wikidata':
            return WikidataScenario(**scenario_conf)
        elif scenario_name == 'wikidata_multi_choice':
            return WikidataMultiChoiceScenario(**scenario_conf)
        elif scenario_name == 'multi_choice':
            return MultiChoiceScenario(**scenario_conf)
        elif scenario_name == 'question_answering':
            return QuestionAnsweringScenario(**scenario_conf)
        else:
            raise ValueError(f'scenario name {scenario_name} is not valid')
