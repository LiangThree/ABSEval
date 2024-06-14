import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ipdb
import json
import yaml
import argparse
import pprint
from pathlib import Path
from typing import List, Dict, Union
from copy import deepcopy

from llm_eval.runner import Runner, RunSpec
from llm_eval.data.request import request_metric_to_dict
from llm_eval.metrics.metric import MetricSpec
        

def expand_run_specs_by_models(run_spec: RunSpec) -> List[RunSpec]:
    model_confs = run_spec.model_conf
    if not isinstance(model_confs, list):
        return run_spec
    
    run_specs: List[RunSpec] = []
    for model_conf in model_confs:
        run_spec_copy = deepcopy(run_spec)
        run_spec_copy.model_conf = model_conf
        run_specs.append(run_spec_copy)
    return run_specs


def expand_run_specs_by_target_views(run_spec: RunSpec) -> List[RunSpec]:
    category = run_spec.scenario_conf['category']
    if not isinstance(category, list):
        return [run_spec]

    run_specs: List[RunSpec] = []
    for one_category in category:
        run_spec_copy = deepcopy(run_spec)
        run_spec_copy.scenario_conf['category'] = one_category
        run_specs.append(run_spec_copy)
    return run_specs


def load_model_conf(
        model_repo_id: str, 
        model_conf_path: str, 
        acceleration_method: str
    ) -> Union[Dict, List[Dict]]:
    """
    从model_conf_path指向的文件中读取模型配置信息，并返回
    如果 model_repo_id 是列表，返回一个 model_conf 列表：List[Dict]
    """
    if isinstance(model_repo_id, list):
        model_repo_ids = model_repo_id
        model_confs = []
        for model_repo_id in model_repo_ids:
            model_conf = load_model_conf(model_repo_id, model_conf_path, acceleration_method)
            model_confs.append(model_conf)
        return model_confs
    elif isinstance(model_repo_id, str):
        # config = yaml.load(open(model_conf_path, 'r'), Loader=yaml.FullLoader)
        model_org, model_name = model_repo_id.split('/')
        # model_conf = config[model_org][model_name]
        model_conf = {
            'model_org': model_org,
            'model_name': model_name,
            'model_repo_id': model_repo_id,
            'model_config_path': model_conf_path,
            'acceleration_method': acceleration_method,
        }
        return model_conf
    else:
        raise ValueError(f'invalid model_conf_path: {model_conf_path}')
    
    
def create_metric_config(run_conf) -> MetricSpec:
    """
    利用run_conf生成 MetricSpec
    """
    model_conf = {
        'acceleration_method': run_conf['scenario_conf']['acceleration_method'],
        'model_repo_id': run_conf['metric_conf']['model_repo_id'],
        'model_conf_path': run_conf['model_conf_path']
    }
    metric_name = run_conf['metric_conf']['metric_name']
    eval_prompt_format = run_conf['metric_conf']['eval_prompt_format']
    metric_spec = MetricSpec(
        metric_name=metric_name, 
        model_conf=model_conf, 
        eval_prompt_format=eval_prompt_format)
    return metric_spec
    

def read_run_specs(path: str) -> List[RunSpec]:
    """
    读取path指向文件中的配置信息，并解析成RunSpec对象；
    这些对象是Runner.run_one()方法的输入
    """
    with open(path) as f:
        run_confs = json.load(f)

    # 将run_confs解析成run_specs
    run_specs: List[RunSpec] = []
    for run_conf in run_confs:
        db_path = Path(run_conf['db_path'])
        scenario_config = run_conf['scenario_conf']
        adapter_method = run_conf['adapter_method']
        model_repo_id = run_conf['model_repo_id']
        metric_spec: MetricSpec = create_metric_config(run_conf)
        acceleration_method = scenario_config['acceleration_method']
        model_conf = load_model_conf(model_repo_id, run_conf['model_conf_path'], acceleration_method)
        run_spec = RunSpec(db_path, scenario_config, adapter_method, model_conf, metric_spec)
        run_specs.append(run_spec)
    
    # 将run_specs扩展成多个，每个run_spec中只有一个model_repo_id
    expand_run_specs: List[RunSpec] = []
    for run_spec in run_specs:
        expand_run_specs.extend(expand_run_specs_by_models(run_spec))
    run_specs = expand_run_specs
    
    # 将run_specs扩展成多个，每个run_spec中只有一个target_view
    expand_run_specs: List[RunSpec] = []
    for run_spec in run_specs:
        expand_run_specs.extend(expand_run_specs_by_target_views(run_spec))
    run_specs = expand_run_specs

    return run_specs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-specs', type=str, default='config/run_specs.json')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--num-instances', type=int, default=-1)
    return parser.parse_args()

def main():
    args = get_args()
    run_specs: List[RunSpec] = read_run_specs(args.run_specs)
    runner = Runner()
    for run_spec in run_specs:
        pprint.pprint(run_spec)
        # 结果已经在run_one()中保存到数据库中了
        runner.run_one(run_spec, test=args.test)

if __name__ == "__main__":
    main()