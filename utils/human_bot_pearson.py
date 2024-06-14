"""
本模块用于计算人类和大模型评测结果的相关系数
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ipdb
import json
import sqlite3
from llm_eval.metrics.pearson_metric import PearsonMetric
from collections import defaultdict
import argparse

def human_bot_pearson(db_path):
    pearson = PearsonMetric()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    human_eval_rows = cursor.execute('select * from human_eval').fetchall()
    eval_models = cursor.execute('select distinct eval_model_name from eval_result').fetchall()
    eval_models = [row[0] for row in eval_models]
    for eval_model in eval_models:
        human_eval, model_eval = [], []
        counter = defaultdict(int)
        for row in human_eval_rows:
            question_id = str(row[0])
            inference_model_name = row[1]
            model_eval_row = cursor.execute('select * from eval_result where question_id=? and model_name=? and eval_model_name=?', (question_id, inference_model_name, eval_model,)).fetchone()
            if model_eval_row is None:
                continue
            # get float model_result from model_result_row
            model_result = model_eval_row[3]
            model_result = float(json.loads(model_result)['answer'])
            # get float human_result from human_result_row
            human_result = row[2]
            human_result = 1.0 if human_result == 'correct' else 0.0
            # append to list
            human_eval.append(human_result)
            model_eval.append(model_result)
        print(eval_model)
        print('pearson  value:', pearson(human_eval, model_eval))


def split_by_eval_model(eval_result_rows):
    """
    将eval_result_rows按照model进行划分
    :param eval_result_rows:
    :return:
    """
    model_to_eval_result_rows = defaultdict(list)
    for row in eval_result_rows:
        model = row[2]
        model_to_eval_result_rows[model].append(row)
    return model_to_eval_result_rows
    

def count_model_eval(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # count model eval
    result = cursor.execute('select eval_model_name, COUNT(*) from eval_result group by eval_model_name').fetchall()
    print(result)
    result = cursor.execute('select * from eval_result').fetchall()
    print(len(result))
    result = cursor.execute('select * from human_eval').fetchall()
    print(len(result))
        
        
if __name__ == "__main__":
    db_path = 'data/database/language_test.db'
    human_bot_pearson('data/database/language_test.db')
    # count_model_eval(db_path)