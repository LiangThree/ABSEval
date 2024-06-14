import sqlite3
import openai
import json
from llm_eval.proxy.services.service import Service
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
                FROM gpt_eval AS ge
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
        cursor = self.conn.cursor()
        cursor.execute(query, (score, question_id, model_name))
        self.conn.commit()

    def insert_into_eval_result(self, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain):
        cursor = self.conn.cursor()
        query = '''INSERT INTO gpt_eval (question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        cursor.execute(query, (question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain))
        self.conn.commit()

def chat(prompt):
    llm_service = LLMServiceFactory.get_llm_service(
        model_repo_id="qwen/Qwen-110B-Chat",
        model_config_path="config/model_config_docker.yaml",
        acceleration_method="vllm"
    )
    response = llm_service.chat(prompt)
    return response[0]


def get_qwen_eval():
    db = EvalDataBase("data/database/script.db")
    questions = db.select_human_eval_question()
    for question in questions:
        question_id, model_name, inference = question
        question, limitation = db.get_question(question_id)
        path = "data/metrics/gpt_eval_prompt.txt"
        with open(path, encoding='utf-8') as f:
            eval_prompt_format = f.read()
        prompt = eval_prompt_format.replace('QUESTION', question)
        prompt = prompt.replace('LIMITATION', limitation)
        prompt = prompt.replace('MODEL_INFERENCE', inference)


        answer = chat(prompt)
        eval_result = json.loads(answer)
        missing_steps = eval_result['missing_steps']
        redundant_steps = eval_result['redundant_steps']
        duplicate_steps = eval_result['duplicate_steps']
        executable = eval_result['executable']
        limitation = eval_result['meet_limitation']
        complete = eval_result['complete_goal']
        step_order = eval_result['step_order_correct']
        explain = eval_result['explain']

        score = 0
        if missing_steps == 'False':
            score += 2
        if redundant_steps == 'False':
            score += 2
        if duplicate_steps == 'False':
            score += 2
        if executable == 'True':
            score += 1
        if limitation == 'True':
            score += 3
        if complete == 'True':
            score += 3
        if step_order == 'True':
            score += 2

        print(score)
        print(question_id, model_name, missing_steps, redundant_steps, duplicate_steps,
              executable, limitation, complete, step_order, explain)
        print('---------------------------------------------------')
        db.write_score(question_id, model_name, 'Qwen-110B-Chat', score)
        db.insert_into_eval_result(question_id, model_name, missing_steps, redundant_steps, duplicate_steps,
                                   executable, limitation, complete, step_order, explain)


if __name__ == '__main__':
    get_qwen_eval()