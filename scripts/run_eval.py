import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run import get_args, read_run_specs
from typing import List
from pprint import pprint
import argparse
import json
from dataclasses import dataclass
from llm_eval.eval_runner import EvalRunSpec, MetricRunner
from llm_eval.metrics.metric import MetricSpec
    

def read_eval_run_specs(eval_specs_path: str, follow: str) -> List[EvalRunSpec]:
    with open(eval_specs_path, 'r') as f:
        eval_specs = json.load(f)
        
    eval_run_specs: List[EvalRunSpec] = []
    for eval_spec in eval_specs:
        db_path = eval_spec['db_path']
        category = eval_spec['category']
        metric_name = eval_spec['metric_conf']['metric_name']
        inference_model_repo_id = eval_spec['inference_model_repo_id']
        model_list = eval_spec['metric_conf']['model_repo_id']
        if not follow:
            if 'ALL' in category:
                eval_model = model_list[0]
                model_conf = {
                    "model_repo_id": eval_model,
                    "acceleration_method": eval_spec['metric_conf']['acceleration_method'],
                    "model_conf_path": eval_spec['model_conf_path'],
                }
                eval_prompt_format = eval_spec['metric_conf']['eval_prompt_format']
                metric_spec: MetricSpec = MetricSpec(metric_name, model_conf, eval_prompt_format)
                eval_run_spec = EvalRunSpec(db_path, 'ALL', inference_model_repo_id, metric_spec)
                eval_run_specs.append(eval_run_spec)
            else:
                model_conf = {
                    "model_repo_id": model_list[0],
                    "acceleration_method": eval_spec['metric_conf']['acceleration_method'],
                    "model_conf_path": eval_spec['model_conf_path'],
                }
                eval_prompt_format = eval_spec['metric_conf']['eval_prompt_format']
                metric_spec: MetricSpec = MetricSpec(metric_name, model_conf, eval_prompt_format)
                eval_run_spec = EvalRunSpec(db_path, category, inference_model_repo_id, metric_spec)
                eval_run_specs.append(eval_run_spec)
        else:
            for one_category in category:
                for eval_model in model_list:
                    if follow and eval_model == model_list[0]:
                        continue
                    model_conf = {
                        "model_repo_id": eval_model,
                        "acceleration_method": eval_spec['metric_conf']['acceleration_method'],
                        "model_conf_path": eval_spec['model_conf_path'],
                    }
                    eval_prompt_format = eval_spec['metric_conf']['eval_prompt_format']
                    metric_spec: MetricSpec = MetricSpec(metric_name, model_conf, eval_prompt_format)
                    eval_run_spec = EvalRunSpec(db_path, one_category, inference_model_repo_id, metric_spec)
                    eval_run_specs.append(eval_run_spec)
    return eval_run_specs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-specs', type=str, default='config/run_eval_specs.json')
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--num-instances', type=int, default=-1)
    parser.add_argument('--annotated', action='store_true')
    parser.add_argument('--interfere', action='store_true')
    return parser.parse_args()


def main():
    # load args
    args = get_args()
    
    # load run_specs
    run_specs: List[EvalRunSpec] = read_eval_run_specs(args.run_specs, args.follow)
    
    metric_runner: MetricRunner = MetricRunner()
    for run_spec in run_specs:
        from colorama import Fore
        print(Fore.BLUE + f'---------------------------------- config ----------------------------------')
        pprint(run_spec)
        print(f'---------------------------------- end ----------------------------------' + Fore.RESET)
        print(Fore.RED + f'---------------------------------- config ----------------------------------')
        pprint(args)
        print(f'---------------------------------- end ----------------------------------' + Fore.RESET)
        metric_runner.run_metric(run_spec, args)
        

if __name__ == "__main__":
    main()