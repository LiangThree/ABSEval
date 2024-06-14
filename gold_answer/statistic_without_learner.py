import sqlite3
from proplot import rc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def select_data_without_learn(self):
        query = """
            SELECT * FROM gold_answer_without_learn
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def select_one_data_from_eval_result(self, eval_model, question_id, model_name):
        query = """
            SELECT  missing_steps, redundant_steps, duplicate_steps FROM eval_result
            WHERE eval_model_name = ? AND question_id = ? AND model_name = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (eval_model, question_id, model_name))
        result = cursor.fetchone()
        return result

    def select_one_data_from_human_eval(self, question_id):
        query = """
            SELECT model_name, missing_steps, redundant_steps, duplicate_steps FROM human_eval
            WHERE question_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id, ))
        result = cursor.fetchone()
        return result





if __name__ == '__main__':
    eval_db = EvalDataBase('data/database/script.db')
    without_learn_data = eval_db.select_data_without_learn()

    without_learn_missing_steps_count = 0
    without_learn_redundant_steps_count = 0
    without_learn_duplicate_steps_count = 0

    maseval_missing_steps_count = 0
    maseval_redundant_steps_count = 0
    maseval_duplicate_steps_count = 0

    for one_data in tqdm(without_learn_data):
        model_name, question_id, answer, missing_steps, redundant_steps, duplicate_steps = one_data
        human_eval = eval_db.select_one_data_from_human_eval(question_id)
        maseval = eval_db.select_one_data_from_eval_result(model_name, question_id, human_eval[0])

        gold_answer = human_eval[1:]
        if missing_steps == gold_answer[0]:
            without_learn_missing_steps_count += 1
        if redundant_steps == gold_answer[1]:
            without_learn_redundant_steps_count += 1
        if duplicate_steps == gold_answer[2]:
            without_learn_duplicate_steps_count += 1


        if maseval[0] == gold_answer[0]:
            maseval_missing_steps_count += 1
        if maseval[1] == gold_answer[1]:
            maseval_redundant_steps_count += 1
        if maseval[2] == gold_answer[2]:
            maseval_duplicate_steps_count += 1


    print("|Whether learn|Missing steps|Redundant steps|Duplicate steps|")
    print("|:--:|:--:|:--:|:--:|")
    print(f"|With Answer Synthesize|{round((maseval_missing_steps_count)/len(without_learn_data), 3)}|{round((maseval_redundant_steps_count)/len(without_learn_data), 3)}|{round((maseval_duplicate_steps_count)/len(without_learn_data), 3)}|")
    print(f"|Without Answer Synthesize|{round((without_learn_missing_steps_count)/len(without_learn_data), 3)}|{round((without_learn_redundant_steps_count)/len(without_learn_data), 3)}|{round((without_learn_duplicate_steps_count)/len(without_learn_data), 3)}|")
