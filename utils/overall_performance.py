import numpy as np
import matplotlib.pyplot as plt
import json
import sqlite3
import random
import os
from pprint import pprint

import yaml
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from proplot import rc
import numpy as np
import matplotlib.pyplot as plt


class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def get_model(self):
        query = """
            SELECT distinct model_name FROM eval_result WHERE eval_model_name = 'Qwen-110B-Chat'
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        result = [one[0] for one in result]
        return result

    def get_eval_result(self):
        query = """
            SELECT * FROM eval_result
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result


eval_db = EvalDataBase('data/database/script.db')

def get_eval_data():
    model_name = eval_db.get_model()
    model_name.append('ALL')
    print(model_name)
    eval_result = eval_db.get_eval_result()
    model_dict = {}
    for model in model_name:
        model_dict[model] = {}
        model_dict[model]['missing_steps'] = 0
        model_dict[model]['redundant_steps'] = 0
        model_dict[model]['duplicate_steps'] = 0
        model_dict[model]['executable'] = 0
        model_dict[model]['limitation'] = 0
        model_dict[model]['complete'] = 0
        model_dict[model]['order'] = 0
        model_dict[model]['count'] = 0
        model_dict[model]['acc'] = 0


    for one_eval_result in tqdm(eval_result, desc='read eval result:'):
        eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, order, _, _ = one_eval_result
        if missing_steps == 'False':
            model_dict[model_name]['missing_steps'] += 1
        if redundant_steps == 'False':
            model_dict[model_name]['redundant_steps'] += 1
        if duplicate_steps == 'False':
            model_dict[model_name]['duplicate_steps'] += 1
        if executable == 'True':
            model_dict[model_name]['executable'] += 1
        if limitation == 'True':
            model_dict[model_name]['limitation'] += 1
        if complete == 'True':
            model_dict[model_name]['complete'] += 1
        if order == 'True':
            model_dict[model_name]['order'] += 1
        model_dict[model_name]['count'] += 1





    print('|model name|missing steps|redundant steps|duplicate steps|executable|limitation|complete|order|')
    print('|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|')
    for model_name in model_dict.keys():
        model_dict['ALL']['missing_steps'] += model_dict[model_name]['missing_steps']
        model_dict['ALL']['redundant_steps'] += model_dict[model_name]['redundant_steps']
        model_dict['ALL']['duplicate_steps'] += model_dict[model_name]['duplicate_steps']
        model_dict['ALL']['executable'] += model_dict[model_name]['executable']
        model_dict['ALL']['limitation'] += model_dict[model_name]['limitation']
        model_dict['ALL']['complete'] += model_dict[model_name]['complete']
        model_dict['ALL']['order'] += model_dict[model_name]['order']
        model_dict['ALL']['count'] += model_dict[model_name]['count']

        model_dict[model_name]['missing_steps'] = model_dict[model_name]['missing_steps'] / model_dict[model_name][
            'count']
        model_dict[model_name]['redundant_steps'] = model_dict[model_name]['redundant_steps'] / model_dict[model_name][
            'count']
        model_dict[model_name]['duplicate_steps'] = model_dict[model_name]['duplicate_steps'] / model_dict[model_name][
            'count']
        model_dict[model_name]['executable'] = model_dict[model_name]['executable'] / model_dict[model_name]['count']
        model_dict[model_name]['limitation'] = model_dict[model_name]['limitation'] / model_dict[model_name]['count']
        model_dict[model_name]['complete'] = model_dict[model_name]['complete'] / model_dict[model_name]['count']
        model_dict[model_name]['order'] = model_dict[model_name]['order'] / model_dict[model_name]['count']

        print(f'|{model_name}|{model_dict[model_name]["missing_steps"]:.3f}|{model_dict[model_name]["redundant_steps"]:.3f}|{model_dict[model_name]["duplicate_steps"]:.3f}|{model_dict[model_name]["executable"]:.3f}|{model_dict[model_name]["limitation"]:.3f}|{model_dict[model_name]["complete"]:.3f}|{model_dict[model_name]["order"]:.3f}|')


if __name__ == '__main__':
    get_eval_data()