import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List
from pprint import pprint
import argparse
import json
from scripts.run import get_args, read_run_specs
from llm_eval.runner import InferenceRunner, RunSpec
from typing import List
from pprint import pprint



def main():
    # load args
    args = get_args()
    
    # load run_specs
    run_specs: List[RunSpec] = read_run_specs(args.run_specs)
    
    metric_runner: InferenceRunner = InferenceRunner()
    for run_spec in run_specs:
        from colorama import Fore
        print(Fore.BLUE)
        pprint(run_spec)
        print(Fore.RESET)
        metric_runner.run_inference(run_spec, args.test)
        

if __name__ == "__main__":
    main()