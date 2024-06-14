import sqlite3
from typing import List

import openai
import json
from tqdm import *
from proxy.services.service import Service
from proxy.services.llm_service_factory import LLMServiceFactory

class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def select_model_rank_question(self):
        query = """
            SELECT question_id, model_name, inference
            FROM llm_inference AS li
            WHERE EXISTS(
                SELECT 1
                FROM model_rank AS mr
                WHERE li.question_id = mr.question_id AND li.model_name = mr.model_name AND mr.Qwen is null )
            AND NOT EXISTS(
                SELECT 1 
                FROM model_eval me 
                WHERE li.question_id = me.question_id AND li.model_name = me.model_name
            )
            """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def get_human_eval_llm_inference(self):
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

    def select_human_eval_question(self):
        query = """
            SELECT question_id, model_name, inference
            FROM llm_inference AS li
            WHERE EXISTS(
                SELECT 1
                FROM human_eval AS he
                WHERE li.question_id = he.question_id AND li.model_name = he.model_name
            ) AND NOT EXISTS(
                SELECT 1
                FROM model_eval AS ge
                WHERE li.question_id = ge.question_id AND li.model_name = ge.model_name
            )
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def get_question(self, question_id):
        query = """
            SELECT question, limitation
            FROM question
            WHERE question_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id,))
        result = cursor.fetchone()
        return result

    def write_score(self, question_id, model_name, evaluator, score):
        query = f"""
            UPDATE eval_score 
            SET {evaluator}= ?
            WHERE question_id = ? AND model_name = ?
        """
        print(query)
        cursor = self.conn.cursor()
        cursor.execute(query, (score, question_id, model_name))
        self.conn.commit()

    def insert_into_eval_result(self, question_id, model_name, eval_model, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain):
        cursor = self.conn.cursor()
        query = '''INSERT INTO model_eval (question_id, model_name, eval_model, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        cursor.execute(query, (question_id, model_name, eval_model, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain))
        self.conn.commit()

    def select_executable_question(self):
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

    def get_gold_answer_without_learn(self, question_id):
        query = """
            SELECT answer
            FROM gold_answer_without_learn
            WHERE question_id = ?
         """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id,))
        result = cursor.fetchone()
        return result

    def update_gold_answer_without_learn(self, question_id, missing_steps, redundant_steps, duplicate_steps):
        cursor = self.conn.cursor()
        query = '''UPDATE gold_answer_without_learn SET missing_steps = ?, redundant_steps = ?, duplicate_steps = ? WHERE question_id = ?'''
        cursor.execute(query, (missing_steps, redundant_steps, duplicate_steps, question_id))
        self.conn.commit()

def load_model():
    llm_service = LLMServiceFactory.get_llm_service(
        model_repo_id="qwen/Qwen-110B-Chat",
        model_config_path="config/model_config_docker.yaml",
        acceleration_method="vllm"
    )
    return llm_service

def chat(llm_service, prompt):
    response = llm_service.chat(prompt)
    return response[0]

def get_qwen_eval():
    llm_service = load_model()
    db = EvalDataBase("data/database/script.db")
    # questions = db.select_human_eval_question()
    questions = db.select_model_rank_question()
    for question in tqdm(questions):
        question_id, model_name, inference = question
        question, limitation = db.get_question(question_id)
        path = "data/metrics/gpt_eval_prompt.txt"
        with open(path, encoding='utf-8') as f:
            eval_prompt_format = f.read()
        prompt = eval_prompt_format.replace('QUESTION', question)
        prompt = prompt.replace('LIMITATION', limitation)
        prompt = prompt.replace('MODEL_INFERENCE', inference)

        answer = chat(llm_service, prompt)

        try:
            eval_result = json.loads(answer)
        except:
            continue

        missing_steps = eval_result['missing_steps']
        redundant_steps = eval_result['redundant_steps']
        duplicate_steps = eval_result['duplicate_steps']
        executable = eval_result['executable']
        limitation = eval_result['meet_limitation']
        complete = eval_result['complete_goal']
        step_order = eval_result['step_order_correct']
        # explain = eval_result['explain']

        if 'false' in missing_steps.lower():
            missing_steps = 'False'
        else:
            missing_steps = 'True'

        if 'false' in redundant_steps.lower():
            redundant_steps = 'False'
        else:
            redundant_steps = 'True'

        if 'false' in duplicate_steps.lower():
            duplicate_steps = 'False'
        else:
            duplicate_steps = 'True'

        if 'true' in executable.lower():
            executable = 'True'
        else:
            executable = 'False'

        if 'true' in limitation.lower():
            limitation = 'True'
        else:
            limitation = 'False'

        if 'true' in complete.lower():
            complete = 'True'
        else:
            complete = 'False'

        if 'true' in step_order.lower():
            step_order = 'True'
        else:
            step_order = 'False'

        explain = None
        print(question_id, model_name, missing_steps, redundant_steps, duplicate_steps,
              executable, limitation, complete, step_order, explain)
        print('---------------------------------------------------')
        # db.write_score(question_id, model_name, 'Qwen-110B-Chat', score)
        db.insert_into_eval_result(question_id, model_name, 'Qwen-110B-Chat', missing_steps, redundant_steps, duplicate_steps,
                                   executable, limitation, complete, step_order, explain)


def get_executable_eval():
    llm_service = load_model()
    eval_db = EvalDataBase("data/database/script.db")
    rows = eval_db.select_executable_question()
    for i in tqdm(range(len(rows))):
        item = rows[i]
        eval_model_name = item[0]
        question_id = item[1]
        model_name = item[2]
        inference = eval_db.select_inference_from_llm_inference(question_id, model_name)
        inference = inference[0][2]

        with open("data/metrics/commonsense_prompt.txt", encoding='utf-8') as f:
            prompt = f.read()
        prompt = prompt.replace('MODEL_INFERENCE', inference)
        answer = chat(llm_service, prompt)

        if 'true' in answer.lower():
            eval_db.update_executable_of_eval_result('True', eval_model_name, question_id, model_name)
        elif 'false' in answer.lower():
            eval_db.update_executable_of_eval_result('False', eval_model_name, question_id, model_name)
        else:
            print("invalid answer:")
            print(answer)

def eval_without_learn_goal_answer():
    llm_service = load_model()
    eval_db = EvalDataBase("data/database/script.db")
    human_eval = eval_db.get_human_eval_llm_inference()

    for one_inference in tqdm(human_eval):
        question_id, model_name, inference = one_inference
        question = eval_db.get_question(question_id)[0]
        gold_answer = eval_db.get_gold_answer_without_learn(question_id)[0]
        with open("data/metrics/eval_prompt.txt", encoding='utf-8') as f:
            prompt = f.read()
        prompt = prompt.replace('Question', question)
        prompt = prompt.replace('Gold Answer', gold_answer)
        prompt = prompt.replace('Model Answer', inference)

        answer = chat(llm_service, prompt)
        try:
            eval_result = json.loads(answer)
        except:
            continue

        missing_steps = eval_result['missing_steps']
        redundant_steps = eval_result['redundant_steps']
        duplicate_steps = eval_result['duplicate_steps']

        eval_db.update_gold_answer_without_learn(question_id, missing_steps, redundant_steps, duplicate_steps)





if __name__ == '__main__':
    # get_executable_eval()
    # get_qwen_eval()
    eval_without_learn_goal_answer()