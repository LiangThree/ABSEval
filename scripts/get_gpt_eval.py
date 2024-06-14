import sqlite3
import time

import openai
import json
import httpx
from openai import OpenAI, ChatCompletion
from tqdm import *
openai.api_key = ""

class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def select_human_eval_question(self, eval_model):
        query = f"""
            SELECT question_id, model_name, inference
            FROM llm_inference AS li
            WHERE EXISTS(
                SELECT 1
                FROM human_eval AS he
                WHERE li.question_id = he.question_id AND li.model_name = he.model_name
            ) AND NOT EXISTS(
                SELECT 1
                FROM model_eval AS ge
                WHERE li.question_id = ge.question_id AND li.model_name = ge.model_name AND ge.eval_model = '{eval_model}'
            )
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def check_question_exists(self, question_id, model_name, eval_model_name):
        query = """
                SELECT 1
                FROM model_eval 
                WHERE question_id = ? AND model_name = ? AND eval_model = ?
                """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id, model_name, eval_model_name))
        result = cursor.fetchall()
        if len(result) > 0:
            return True
        return False

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
        cursor = self.conn.cursor()
        cursor.execute(query, (score, question_id, model_name))
        self.conn.commit()

    def insert_into_eval_result(self, question_id, model_name, eval_model, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain):
        cursor = self.conn.cursor()
        query = '''INSERT INTO model_eval (question_id, model_name, eval_model, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        cursor.execute(query, (question_id, model_name, eval_model, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain))
        self.conn.commit()

def chat_gpt_3(prompt):
    client = OpenAI(
        base_url="https://api.xty.app/v1",
        api_key="",
        http_client=httpx.Client(
            base_url="https://api.xty.app/v1",
            follow_redirects=True,
        ),
    )

    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    answer = completion.choices[0].message.content

    start = answer.find('{')  # 找到第一个左大括号的索引
    if start == -1:  # 如果找不到左大括号
        return None
    end = answer.find('}', start)  # 从左大括号的索引开始找右大括号
    if end == -1:  # 如果找不到右大括号
        return None

    return answer[start:end + 1]


def chat_gpt_4(prompt):
    client = OpenAI(
        base_url="https://api.xty.app/v1",
        api_key="",
        http_client=httpx.Client(
            base_url="https://api.xty.app/v1",
            follow_redirects=True,
        ),
    )

    completion = client.chat.completions.create(
        model='gpt-4-turbo',
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    answer = completion.choices[0].message.content

    start = answer.find('{')  # 找到第一个左大括号的索引
    if start == -1:  # 如果找不到左大括号
        return None
    end = answer.find('}', start)  # 从左大括号的索引开始找右大括号
    if end == -1:  # 如果找不到右大括号
        return None

    return answer[start:end + 1]


def get_gpt_eval(model):
    db = EvalDataBase("data/database/script.db")
    if model == 3:
        questions = db.select_human_eval_question('gpt-3.5-turbo')
    elif model == 4:
        questions = db.select_human_eval_question('gpt-4-turbo')
    for question in tqdm(questions):
        question_id, model_name, inference = question
        question, limitation = db.get_question(question_id)
        path = "data/metrics/gpt_eval_prompt.txt"
        with open(path, encoding='utf-8') as f:
            eval_prompt_format = f.read()
        prompt = eval_prompt_format.replace('QUESTION', question)
        prompt = prompt.replace('LIMITATION', limitation)
        prompt = prompt.replace('MODEL_INFERENCE', inference)

        if model == 3:
            check = db.check_question_exists(question_id, model_name, 'gpt-3.5-turbo')
        elif model == 4:
            check = db.check_question_exists(question_id, model_name, 'gpt-4-turbo')
        else:
            print("no model!")
            return

        if check:
            print('eval exists')
            continue

        if model == 3:
            answer = chat_gpt_3(prompt)
        elif model == 4:
            answer = chat_gpt_4(prompt)

        try:
            eval_result = json.loads(answer)
        except Exception:
            print(Exception)
            print(answer)
            continue

        try:
            missing_steps = eval_result['missing_steps']
            redundant_steps = eval_result['redundant_steps']
            duplicate_steps = eval_result['duplicate_steps']
            executable = eval_result['executable']
            limitation = eval_result['meet_limitation']
            complete = eval_result['complete_goal']
            step_order = eval_result['step_order_correct']
            explain = ""
        except Exception:
            print(Exception)
            print(answer)
            continue

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


        print(question_id, model_name, missing_steps, redundant_steps, duplicate_steps,
              executable, limitation, complete, step_order, explain)
        print('---------------------------------------------------')
        if model == 3:
            db.insert_into_eval_result(question_id, model_name, 'gpt-3.5-turbo', missing_steps, redundant_steps,
                                       duplicate_steps,
                                       executable, limitation, complete, step_order, explain)
        elif model == 4:
            db.insert_into_eval_result(question_id, model_name, 'gpt-4-turbo', missing_steps, redundant_steps,
                                       duplicate_steps,
                                       executable, limitation, complete, step_order, explain)


if __name__ == '__main__':

    while True:
        db = EvalDataBase("data/database/script.db")
        questions = db.select_human_eval_question('gpt-4-turbo')
        if len(questions) > 0:
            try:
                get_gpt_eval(4)
            except Exception:
                print(Exception)
                time.sleep(3)
                continue
        else:
            exit(0)
