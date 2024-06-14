import sqlite3
import json
from tqdm import *
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from llm_eval.proxy.services.llm_service_factory import LLMServiceFactory

class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

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
                FROM whether_gold_answer AS ge
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


    def insert_into_whether_gold_answer(self, question_id, model_name, with_gold_answer, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order):
        cursor = self.conn.cursor()
        query = '''INSERT INTO whether_gold_answer (question_id, model_name, with_gold_answer, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        cursor.execute(query, (question_id, model_name, with_gold_answer, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order))
        self.conn.commit()

    def get_gold_answer(self, question_id):
        query = """
            SELECT answer
            FROM gold_answer
            WHERE question_id = ? AND model_name = 'Qwen-110B-Chat'
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id,))
        result = cursor.fetchone()
        return result


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

def deal_answer(answer):
    eval_result = json.loads(answer)

    missing_steps = eval_result['missing_steps']
    redundant_steps = eval_result['redundant_steps']
    duplicate_steps = eval_result['duplicate_steps']
    executable = eval_result['executable']
    limitation = eval_result['meet_limitation']
    complete = eval_result['complete_goal']
    step_order = eval_result['step_order_correct']

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

    return missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order



def get_qwen_eval():
    llm_service = load_model()
    db = EvalDataBase("data/database/script.db")
    questions = db.select_human_eval_question()
    for question in tqdm(questions):
        question_id, model_name, inference = question
        question, question_limitation = db.get_question(question_id)
        gold_answer = db.get_gold_answer(question_id)[0]


        with_answer_path = "gold_answer/with_gold_answer.txt"
        without_answer_path = "gold_answer/without_gold_answer.txt"


        with open(without_answer_path, encoding='utf-8') as f:
            eval_prompt_format = f.read()
        prompt = eval_prompt_format.replace('QUESTION', question)
        prompt = prompt.replace('LIMITATION', question_limitation)
        prompt = prompt.replace('MODEL_INFERENCE', inference)
        answer = chat(llm_service, prompt)

        try:
            missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order = deal_answer(
                answer)
            # print(question_id, model_name, missing_steps, redundant_steps, duplicate_steps,
            #       executable, limitation, complete, step_order)
            db.insert_into_whether_gold_answer(question_id, model_name, 'False', missing_steps, redundant_steps,
                                           duplicate_steps, executable, limitation, complete, step_order)
            # print('---------------------------------------------------')
        except:
            pass



        with open(with_answer_path, encoding='utf-8') as f:
            eval_prompt_format = f.read()
        prompt = eval_prompt_format.replace('QUESTION', question)
        prompt = prompt.replace('LIMITATION', question_limitation)
        prompt = prompt.replace('MODEL_INFERENCE', inference)
        prompt = prompt.replace('GOLD_ANSWER', gold_answer)
        answer = chat(llm_service, prompt)

        try:
            missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order = deal_answer(
                answer)
            # print(question_id, model_name, missing_steps, redundant_steps, duplicate_steps,
            #       executable, limitation, complete, step_order)
            db.insert_into_whether_gold_answer(question_id, model_name, 'True', missing_steps, redundant_steps,
                                               duplicate_steps, executable, limitation, complete, step_order)
            # print('---------------------------------------------------')
        except:
            pass


if __name__ == '__main__':
    get_qwen_eval()