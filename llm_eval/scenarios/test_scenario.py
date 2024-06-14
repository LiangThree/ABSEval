import ipdb
from .scenario_factory import ScenarioFactory


def test_wikidata_multi_choice_scenario():
    scenario_name = 'wikidata_multi_choice'
    scenario_factory = ScenarioFactory()
    scenario = scenario_factory.get_scenario(scenario_name)
    instances = scenario.get_instances()
    ipdb.set_trace()


def main():
    test_wikidata_multi_choice_scenario()


if __name__ == "__main__":
    main()
