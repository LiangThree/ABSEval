import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run import get_args, read_run_specs
from llm_eval.runner import RunSpec, LearnerRunner
from typing import List
from pprint import pprint
import argparse
import json
from dataclasses import dataclass
from llm_eval.runner import EvalRunSpec, LearnRunSpec
from llm_eval.metrics.metric import MetricSpec
from llm_eval.learner.learner import LearnerSpec
from llm_eval.clean_data import filter_gold_answer



def read_learner_run_specs(eval_specs_path: str) -> List[EvalRunSpec]:
    with open(eval_specs_path, 'r') as f:
        eval_specs = json.load(f)

    # eval_run_specs: List[EvalRunSpec] = []
    for eval_spec in eval_specs:
        db_path = eval_spec['db_path']
        category = eval_spec['category']
        metric_name = eval_spec['metric_conf']['metric_name']
        inference_model_repo_id = eval_spec['inference_model_repo_id']
        model_conf = {
            "model_repo_id": eval_spec['metric_conf']['model_repo_id'],
            "acceleration_method": eval_spec['metric_conf']['acceleration_method'],
            "model_conf_path": eval_spec['model_conf_path'],
        }
        learner_prompt_format = eval_spec['metric_conf']['learner_prompt_format']

        expand_run_specs: List[LearnRunSpec] = []
        if category == 'ALL':
            learnerSpec: LearnerSpec = LearnerSpec(metric_name, model_conf, learner_prompt_format)
            eval_run_spec = LearnRunSpec(db_path, 'ALL', inference_model_repo_id, learnerSpec)
            expand_run_specs.append(eval_run_spec)
        else:
            for one_category in category:
                learnerSpec: LearnerSpec = LearnerSpec(metric_name, model_conf, learner_prompt_format)
                eval_run_spec = LearnRunSpec(db_path, one_category, inference_model_repo_id, learnerSpec)
                expand_run_specs.append(eval_run_spec)
        return expand_run_specs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-specs', type=str, default='config/run_learner_specs.json')
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--num-instances', type=int, default=-1)
    parser.add_argument('--annotated', action='store_true')
    return parser.parse_args()


def main():
    # load args
    args = get_args()

    # load run_specs
    run_specs: List[LearnRunSpec] = read_learner_run_specs(args.run_specs)

    learner_runner: LearnerRunner = LearnerRunner()
    for run_spec in run_specs:
        # filter_gold_answer()
        from colorama import Fore
        print(Fore.BLUE + f'---------------------------------- config ----------------------------------')
        pprint(run_spec)
        print(f'---------------------------------- end ----------------------------------'+Fore.RESET)
        learner_runner.run_learner(run_spec, args)
        filter_gold_answer()




if __name__ == "__main__":
    main()