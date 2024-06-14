import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List
from tqdm import *
import sqlite3
import re
import transformers
import torch



@dataclass
class RunSpec:
    db_path: Path
    scenario_conf: dict
    adapter_method: str
    model_conf: dict
    # metric_spec: MetricSpec

class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def select_question_ids_from_eval_results_for_commonsense(self):
        query = """
            SELECT eval_model_name, question_id, model_name
            FROM eval_result
            WHERE executable is NULL
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def select_inference_from_llm_inference(self, question_id, model_name):
        query = """
            SELECT question_id, model_name, inference
            FROM llm_inference
            WHERE question_id = ? AND model_name = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id, model_name))
        result = cursor.fetchall()
        return result

    def update_executable_of_eval_result(self, executable, eval_model_name, question_id, model_name):
        cursor = self.conn.cursor()
        query = '''UPDATE eval_result SET executable = ? WHERE eval_model_name = ? AND question_id = ? AND model_name = ?'''
        cursor.execute(query, (executable, eval_model_name, question_id, model_name))
        self.conn.commit()


def load_vera():
    tokenizer = transformers.AutoTokenizer.from_pretrained('liujch1998/vera')
    model = transformers.T5EncoderModel.from_pretrained('liujch1998/vera')
    model.D = model.shared.embedding_dim
    linear = torch.nn.Linear(model.D, 1, dtype=model.dtype)
    linear.weight = torch.nn.Parameter(model.shared.weight[32099, :].unsqueeze(0))
    linear.bias = torch.nn.Parameter(model.shared.weight[32098, 0].unsqueeze(0))
    model.eval()
    t = model.shared.weight[32097, 0].item()  # temperature for calibration

    return model, tokenizer, linear, t


def calculate_confidence(model, tokenizer, linear, t, statement):
    input_ids = tokenizer.batch_encode_plus([statement], return_tensors='pt', padding='longest',
                                            truncation='longest_first', max_length=1024).input_ids
    with torch.no_grad():
        output = model(input_ids)
        last_hidden_state = output.last_hidden_state
        hidden = last_hidden_state[0, -1, :]
        logit = linear(hidden).squeeze(-1)
        logit_calibrated = logit / t
        score_calibrated = logit_calibrated.sigmoid()
        score_calibrated = float(score_calibrated)
        # print(score_calibrated)

        return score_calibrated

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-specs', type=str, default='config/run_execute_specs.json')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--num-instances', type=int)
    parser.add_argument('--only-annotated-instance', action='store_true')
    return parser.parse_args()


def run_metric(args, specs):
    # these args are used to filter instances
    test = args.test
    num_instances = args.num_instances
    only_annotated_instance = args.only_annotated_instance

    eval_db: EvalDataBase = EvalDataBase(specs['db_path'])
    rows: List[tuple] = eval_db.select_question_ids_from_eval_results_for_commonsense()


    if num_instances:
        rows = random.sample(rows, num_instances)
    print('question number:', len(rows))

    model, tokenizer, linear, t = load_vera()


    for i in tqdm(range(len(rows))):
        item = rows[i]
        eval_model_name = item[0]
        question_id = item[1]
        model_name = item[2]
        inference = eval_db.select_inference_from_llm_inference(question_id, model_name)
        inference = inference[0][2]

        if inference is None:
            eval_db.update_executable_of_eval_result('False', eval_model_name, question_id, model_name)
            continue

        confidence = calculate_confidence(model, tokenizer, linear, t, inference)
        if confidence >= 0.5:
            eval_db.update_executable_of_eval_result('True', eval_model_name, question_id, model_name)
        else:
            eval_db.update_executable_of_eval_result('False', eval_model_name, question_id, model_name)


if __name__ == '__main__':
    args = get_args()
    specs = {
        'db_path': "data/database/script.db",
        'target_view': ['1']
    }
    run_metric(args, specs)


