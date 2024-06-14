import random
from tqdm import *
import numpy as np
import sqlite3
from bert_score import score
import sqlite3
from transformers import logging
from tqdm import *
from rouge_score import rouge_scorer
"""
每个类别选100个问题，一共1000个问题, MASEval\Qwen\BERTScore\ROUGE
"""

class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def select_question(self):
        query = """
                SELECT DISTINCT category
                FROM question
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        category_list = [one[0] for one in result]

        question_list = []
        for one_category in category_list:
            query = """
                SELECT question_id
                FROM question
                WHERE category = ?
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (one_category,))
            result = cursor.fetchall()
            result_list = []
            result_list.extend([one[0] for one in result])
            random.shuffle(result_list)
            question_list.extend(result_list[:10])

        return question_list

    def get_question_ids(self):
        query = """
            SELECT question_id, model_name
            FROM model_rank
            WHERE Rouge is null
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def get_llm_inference(self, question_id, model_name):
        query = """
            SELECT inference
            FROM llm_inference
            WHERE question_id = ? AND model_name = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id, model_name))
        result = cursor.fetchone()
        return result

    def get_gold_answer(self, question_id):
        query = """
            SELECT answer
            FROM gold_answer
            WHERE question_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id, ))
        result = cursor.fetchone()
        return result

    def write_Score(self, question_id, model_name, col, score):
        query = f"""
                UPDATE model_rank SET {col} = ?
                WHERE question_id = ? AND model_name = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (score, question_id, model_name))
        self.conn.commit()

    def write_MASEval_Score(self, question_id, model_name, score):
        query = """
                INSERT INTO model_rank (question_id, model_name, MASEval)
                VALUES (?, ?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id, model_name, score))
        self.conn.commit()

    def select_MASEval(self, question_id_list):
        for one_question_id in tqdm(question_id_list):
            query = """
                SELECT *
                FROM eval_result
                WHERE eval_model_name = 'Qwen-110B-Chat' AND question_id = ?
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (one_question_id, ))
            result = cursor.fetchall()
            for one_eval_result in result:
                eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, order, explain1, explain2 = one_eval_result
                score = 0
                if missing_steps == 'False':
                    score += 1
                if redundant_steps == 'False':
                    score += 1
                if duplicate_steps == 'False':
                    score += 1
                if executable == 'True':
                    score += 1
                if limitation == 'True':
                    score += 1
                if complete == 'True':
                    score += 1
                if order == 'True':
                    score += 1

                try:
                    self.write_MASEval_Score(question_id, model_name, score)
                except Exception:
                    print(Exception)
                    continue

eval_db = EvalDataBase('data/database/repair.db')


def calculate_bert_score(ref_sentence, hyp_sentence):
    # Calculate BERTScore
    P, R, F1 = score([hyp_sentence], [ref_sentence], lang='en', rescale_with_baseline=True)
    bert_score = F1.item()
    return bert_score


def calculate_rouge_score(ref_sentence, hyp_sentence):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref_sentence, hyp_sentence)

    rougel_f1 = scores['rougeL'].fmeasure

    return rougel_f1

def calculate_Score():
    question_list = eval_db.get_question_ids()
    for question_id, model_name in tqdm(question_list):
        llm_inference = eval_db.get_llm_inference(question_id, model_name)[0]
        gold_answer = eval_db.get_gold_answer(question_id)[0]


        BERTScore = calculate_bert_score(gold_answer, llm_inference)
        BERTScore = round(BERTScore*7, 0)
        eval_db.write_Score(question_id, model_name, 'BERTScore', BERTScore)

        Rouge = calculate_rouge_score(gold_answer, llm_inference)
        Rouge = round(Rouge*7, 0)
        eval_db.write_Score(question_id, model_name, 'Rouge', Rouge)



if __name__ == '__main__':
    # question_list = eval_db.select_question()
    # eval_db.select_MASEval(question_list)
    calculate_Score()