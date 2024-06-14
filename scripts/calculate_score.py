import numpy as np
from bert_score import score
import sqlite3
from transformers import logging
from tqdm import *
from rouge_score import rouge_scorer

class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def select_human_eval_result(self):
        query = """
            SELECT *
            FROM human_eval
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def get_human_eval(self, question_id, model_name):
        query = """
            SELECT *
            FROM human_eval
            WHERE question_id = ? AND model_name = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id, model_name))
        result = cursor.fetchone()
        return result

    def select_our_eval_result(self):
        query = """
            SELECT *
            FROM eval_result
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def select_gpt_eval_result(self):
        query = """
            SELECT *
            FROM gpt_eval
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result


    def select_human_eval_question(self):
        query = """
            SELECT question_id, model_name, inference
            FROM llm_inference AS li
            WHERE EXISTS(
                SELECT 1
                FROM human_eval AS he
                WHERE li.question_id = he.question_id AND li.model_name = he.model_name
            )
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def select_eval_result(self, model_name):
        query = f"""
            SELECT *
            FROM eval_result AS er
            WHERE EXISTS(
                SELECT 1 
                FROM model_eval AS he
                WHERE er.question_id = he.question_id AND er.model_name = he.model_name and er.eval_model_name = '{model_name}'
            )
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def select_gpt3_eval(self):
        query = """
            SELECT *
            FROM model_eval AS ge
            WHERE EXISTS(
                SELECT 1 
                FROM human_eval AS he
                WHERE ge.question_id = he.question_id AND ge.model_name = he.model_name and ge.eval_model = 'gpt-3.5-turbo'
            )
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def select_gpt3_eval(self):
        query = """
            SELECT *
            FROM model_eval AS ge
            WHERE EXISTS(
                SELECT 1 
                FROM human_eval AS he
                WHERE ge.question_id = he.question_id AND ge.model_name = he.model_name and ge.eval_model = 'gpt-3.5-turbo'
            )
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def select_gpt4_eval(self):
        query = """
            SELECT *
            FROM model_eval AS ge
            WHERE EXISTS(
                SELECT 1 
                FROM human_eval AS he
                WHERE ge.question_id = he.question_id AND ge.model_name = he.model_name and ge.eval_model = 'gpt-4-turbo'
            )
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def select_qwen_eval(self):
        query = """
            SELECT *
            FROM model_eval AS ge
            WHERE EXISTS(
                SELECT 1 
                FROM human_eval AS he
                WHERE ge.question_id = he.question_id AND ge.model_name = he.model_name and ge.eval_model = 'Qwen-110B-Chat'
            )
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def select_gold_answer(self, question_id):
        query = """
            SELECT question_id, answer
            FROM gold_answer
            WHERE question_id = ? AND model_name = 'Qwen-110B-Chat'
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id,))
        result = cursor.fetchall()
        return result[0]

    def write_human_score(self, question_id, model_name):
        query = """
            INSERT OR REPLACE INTO eval_score (question_id, model_name)
            VALUES (?, ?)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id, model_name))
        self.conn.commit()

    def write_score(self, question_id, model_name, evaluator, score):
        query = f"""
            UPDATE eval_score 
            SET {evaluator}= ?
            WHERE question_id = ? AND model_name = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (score, question_id, model_name))
        self.conn.commit()

    def get_human_and_score_number(self):
        cursor = self.conn.cursor()

        query = """
            SELECT count(*)
            FROM human_eval
        """
        cursor.execute(query, ())
        result = cursor.fetchall()
        human_eval_number = result[0]

        query = """
            SELECT count(*)
            FROM eval_score
        """
        cursor.execute(query, ())
        result = cursor.fetchall()
        eval_score_number = result[0]

        return human_eval_number, eval_score_number

    def get_eval_score(self):
        cursor = self.conn.cursor()
        query = """
            SELECT *
            FROM eval_score
        """
        cursor.execute(query, ())
        result = cursor.fetchall()

        return result


    def close_connection(self):
        self.conn.close()


db = EvalDataBase('data/database/script.db')

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

def get_bert_score():
    llm_inference = db.select_human_eval_question()
    for infernece in tqdm(llm_inference, desc='BERTScore'):
        question_id, model_name, inference = infernece
        question_id, gold_answer = db.select_gold_answer(question_id)
        if infernece is None or gold_answer is None:
            continue
        bert_score = calculate_bert_score(gold_answer, inference)
        bert_score = round(bert_score, 3)
        db.write_score(question_id, model_name, 'BERTScore', bert_score)

        rouge_score = calculate_rouge_score(gold_answer, inference)
        rouge_score = round(rouge_score, 3)
        db.write_score(question_id, model_name, 'Rouge', rouge_score)


def write_human_evaluation_score():
    human_eval = db.select_human_eval_result()
    for one_human_eval in tqdm(human_eval, desc='write human_eval'):
        question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order = one_human_eval
        db.write_human_score(question_id, model_name)


def get_our_score():
    eval_result = db.select_eval_result('Qwen-110B-Chat')
    for inference in tqdm(eval_result, desc='our'):
        eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, _, _ = inference
        our_eval = [1 if x == 'True' else 0 for x in [missing_steps, redundant_steps, duplicate_steps, limitation, complete, step_order]]
        question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order = db.get_human_eval(question_id, model_name)
        human_eval = [1 if x == 'True' else 0 for x in
                    [missing_steps, redundant_steps, duplicate_steps, limitation, complete, step_order]]

        score = calculate_mse(our_eval, human_eval)
        db.write_score(question_id, model_name, 'ours', score)

    eval_result = db.select_eval_result('gpt-3.5-turbo')
    for inference in tqdm(eval_result, desc='gpt-3.5'):
        eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, _, _ = inference
        our_eval = [1 if x == 'True' else 0 for x in
                    [missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order]]
        question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order = db.get_human_eval(
            question_id, model_name)
        human_eval = [1 if x == 'True' else 0 for x in
                      [missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order]]

        score = calculate_mse(our_eval, human_eval)
        db.write_score(question_id, model_name, 'MASEval_GPT3', score)

    eval_result = db.select_eval_result('gpt-4-turbo')
    for inference in tqdm(eval_result, desc='gpt-4'):
        eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, _, _ = inference
        our_eval = [1 if x == 'True' else 0 for x in
                    [missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order]]
        question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order = db.get_human_eval(
            question_id, model_name)
        human_eval = [1 if x == 'True' else 0 for x in
                      [missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order]]

        score = calculate_mse(our_eval, human_eval)
        db.write_score(question_id, model_name, 'MASEval_GPT4', score)


def get_gpt_score():

    gpt_eval = db.select_gpt3_eval()
    for inference in tqdm(gpt_eval, desc='GPT-3.5'):
        question_id, model_name, eval_model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, _ = inference
        our_eval = [1 if x == 'True' else 0 for x in
                    [missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order]]
        question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order = db.get_human_eval(
            question_id, model_name)
        human_eval = [1 if x == 'True' else 0 for x in
                      [missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order]]

        score = calculate_mse(our_eval, human_eval)
        db.write_score(question_id, model_name, 'GPT3', score)

    gpt_eval = db.select_gpt4_eval()
    for inference in tqdm(gpt_eval, desc='GPT-4'):
        question_id, model_name, eval_model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, _ = inference
        our_eval = [1 if x == 'True' else 0 for x in
                    [missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order]]
        question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order = db.get_human_eval(
            question_id, model_name)
        human_eval = [1 if x == 'True' else 0 for x in
                      [missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order]]

        score = calculate_mse(our_eval, human_eval)
        db.write_score(question_id, model_name, 'GPT4', score)

    gpt_eval = db.select_qwen_eval()
    for inference in tqdm(gpt_eval, desc='Qwen'):
        question_id, model_name, eval_model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, _ = inference
        our_eval = [1 if x == 'True' else 0 for x in
                    [missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order]]
        question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order = db.get_human_eval(
            question_id, model_name)
        human_eval = [1 if x == 'True' else 0 for x in
                      [missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order]]

        score = calculate_mse(our_eval, human_eval)
        db.write_score(question_id, model_name, 'Qwen', score)


def get_score():
    human_eval_number, eval_score_number = db.get_human_and_score_number()
    if human_eval_number == eval_score_number:
        pass
    else:
        write_human_evaluation_score()
    # get_our_score()
    # get_bert_score()
    get_gpt_score()


def calculate_euclidean_distance(list_score_1, list_score_2):
    euclidean_distance = np.sqrt(np.sum(np.square(np.array(list_score_1) - np.array(list_score_2)))) / len(list_score_1)
    return euclidean_distance

def euclidean():
    eval_score = db.get_eval_score()
    BERTScore = [int(item[2]) for item in eval_score]
    ours = [int(item[3]) for item in eval_score]
    human = [int(item[4]) for item in eval_score]
    gpt = [int(item[5]) for item in eval_score]

    print('|Method|Euclidean distance|')
    print('|:------|:-----------------|')

    euclidean_distance = calculate_euclidean_distance(ours, human)
    euclidean_distance = round(euclidean_distance, 3)
    print('|ours|', euclidean_distance, '|')

    euclidean_distance = calculate_euclidean_distance(BERTScore, human)
    euclidean_distance = round(euclidean_distance, 3)
    print('|BERTScore|', euclidean_distance, '|')

    euclidean_distance = calculate_euclidean_distance(gpt, human)
    euclidean_distance = round(euclidean_distance, 3)
    print('|gpt|', euclidean_distance, '|')


def minkowski_distance_bool(x, y, p):
    """
    计算两个布尔向量之间的闵氏距离

    参数：
    x: 第一个布尔向量
    y: 第二个布尔向量
    p: 闵氏距离的指数，当p=1时为曼哈顿距离，p=2时为欧几里得距离，通常p取大于等于1的整数或无穷大

    返回：
    两个向量之间的闵氏距离
    """
    x_bool = np.array([True if val.lower() == 'true' else False for val in x])
    y_bool = np.array([True if val.lower() == 'true' else False for val in y])
    return np.power(np.sum(np.power(np.abs(x_bool.astype(int) - y_bool.astype(int)), p)), 1 / p)

def minkowski_distance():
    human_eval = db.select_human_eval_result()
    human_eval = [one_human_eval[2:9] for one_human_eval in human_eval]

    our_eval = db.select_our_eval_result()
    our_eval = [one_our_eval[2:9] for one_our_eval in our_eval]

    gpt_eval = db.select_gpt_eval_result()
    gpt_eval = [one_gpt_eval[2:9] for one_gpt_eval in gpt_eval]

    print('\n')
    print('|Method|Minkowski distance|')
    print('|:------|:-----------------|')

    sum = 0
    for our, human in zip(our_eval, human_eval):
        sum += minkowski_distance_bool(human, our, 7)
    sum = sum / len(our_eval)
    sum = round(sum, 3)
    print('|ours|', sum, '|')

    sum = 0
    for gpt, human in zip(gpt_eval, human_eval):
        sum += minkowski_distance_bool(human, gpt, 7)
    sum = sum / len(our_eval)
    sum = round(sum, 3)
    print('|gpt|', sum, '|')


def calculate_mse(true_values, predicted_values):
    # 确保输入是numpy数组
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    # 计算MSE
    mse = np.mean((true_values - predicted_values) ** 2)
    mse = round(float(mse), 3)

    return mse


def calculate_distance():
    euclidean()
    minkowski_distance()


def calculate_similarity():
    eval_score = db.get_eval_score()
    MASEval_qwen = 0
    MASEval_gpt3 = 0
    MASEval_gpt4 = 0
    Qwen = 0
    gpt3 = 0
    gpt4 = 0
    for one_eval_score in eval_score:
        try:
            MASEval_qwen += float(one_eval_score[2])
            MASEval_gpt3 += float(one_eval_score[3])
            MASEval_gpt4 += float(one_eval_score[4])
            Qwen += float(one_eval_score[5])
            gpt3 += float(one_eval_score[6])
            gpt4 += float(one_eval_score[7])
        except:
            continue

    MASEval_qwen = round(float(MASEval_qwen / len(eval_score)), 3)
    MASEval_gpt3 = round(float(MASEval_gpt3 / len(eval_score)), 3)
    MASEval_gpt4 = round(float(MASEval_gpt4 / len(eval_score)), 3)
    Qwen = round(float(Qwen / len(eval_score)), 3)
    gpt3 = round(float(gpt3 / len(eval_score)), 3)
    gpt4 = round(float(gpt4 / len(eval_score)), 3)

    print('|Eval Metric|Mechanism｜MSE|')
    print('|:--:|:--:|:--:|')
    print('|MASEval Qwen|Multi-Agent|{}|'.format(MASEval_qwen))
    print('|MASEval GPT3|Multi-Agent|{}|'.format(MASEval_gpt3))
    print('|MASEval GPT4|Multi-Agent|{}|'.format(MASEval_gpt4))
    print('|Qwen|Single-Agent|{}|'.format(Qwen))
    print('|GPT3|Single-Agent|{}|'.format(gpt3))
    print('|GPT4|Single-Agent|{}|'.format(gpt4))




if __name__ == '__main__':
    logging.set_verbosity_warning()
    get_score()
    calculate_similarity()
    # calculate_distance()
