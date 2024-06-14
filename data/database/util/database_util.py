import json
import sqlite3
import random
import os
import yaml
from collections import Counter
from tqdm import tqdm
import re

class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def table_create(self):
        """
            Create tables in the database
        """
        cursor = self.conn.cursor()
        # SQL to create tables with all fields as TEXT (string) type
        create_tables_sql = {
            'question': '''
                CREATE TABLE IF NOT EXISTS question (
                    question_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    target_view TEXT,
                    limitation TEXT,
                    abstract_question_id TEXT,
                    abstract_question TEXT,
                    question TEXT,
                    frequency TEXT,
                    answer_length TEXT
                )
            ''',
            'abstract_question': '''
                CREATE TABLE IF NOT EXISTS abstract_question (
                    question_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    abstract_question TEXT,
                    frequency TEXT,
                    valid TEXT
                )
            ''',
            'llm_inference': '''
                CREATE TABLE IF NOT EXISTS llm_inference (
                    question_id TEXT,
                    model_name TEXT,
                    inference TEXT,
                    PRIMARY KEY (question_id, model_name)
                )
            ''',
            'gold_answer': '''
                    CREATE TABLE IF NOT EXISTS gold_answer (
                        model_name TEXT,
                        question_id TEXT,
                        answer TEXT,
                        PRIMARY KEY (question_id, model_name)
                    )
            ''',
            'gold_answer_without_learn': '''
                    CREATE TABLE IF NOT EXISTS gold_answer_without_learn (
                        model_name TEXT,
                        question_id TEXT,
                        answer TEXT,
                        missing_steps TEXT,
                        redundant_steps TEXT,
                        duplicate_steps TEXT,
                        PRIMARY KEY (question_id, model_name)
                    )
            ''',
            'eval_result': '''
                    CREATE TABLE IF NOT EXISTS eval_result (
                       eval_model_name TEXT,
                       question_id TEXT,
                       model_name TEXT,
                       missing_steps TEXT,
                       redundant_steps TEXT,
                       duplicate_steps TEXT,
                       executable TEXT,
                       limitation TEXT,
                       complete TEXT,
                       step_order TEXT,
                       explain_1 TEXT,
                       explain_2 TEXT,
                       PRIMARY KEY (eval_model_name, question_id, model_name)
            )
            ''',
            'human_eval': '''
                    CREATE TABLE IF NOT EXISTS human_eval (
                        question_id TEXT,
                        model_name TEXT,
                        missing_steps TEXT,
                        redundant_steps TEXT,
                        duplicate_steps TEXT,
                        executable TEXT,
                        limitation TEXT,
                        complete TEXT,
                        step_order TEXT,
                        PRIMARY KEY (question_id, model_name)
                    )
                    ''',
            'interfere': '''
                    CREATE TABLE IF NOT EXISTS interfere (
                        question_id TEXT,
                        model_name TEXT,
                        interfere_category TEXT,
                        abstract_question TEXT,
                        question TEXT,
                        question_limitation TEXT,
                        inference TEXT,
                        human_edit TEXT,
                        missing_steps TEXT,
                        redundant_steps TEXT,
                        duplicate_steps TEXT,
                        executable TEXT,
                        limitation TEXT,
                        complete TEXT,
                        step_order TEXT,
                        explain_1 TEXT,
                        explain_2 TEXT,
                        PRIMARY KEY (question_id, model_name, interfere_category)
                    )
            ''',
            'eval_score': '''
                    CREATE TABLE IF NOT EXISTS eval_score (
                        question_id TEXT,
                        model_name TEXT,
                        evaluator TEXT,
                        score TEXT,
                        PRIMARY KEY (question_id, model_name)
                    )
            ''',
            'gpt_eval': '''
                    CREATE TABLE IF NOT EXISTS gpt_eval (
                        question_id TEXT,
                        model_name TEXT,
                        missing_steps TEXT,
                        redundant_steps TEXT,
                        duplicate_steps TEXT,
                        executable TEXT,
                        limitation TEXT,
                        complete TEXT,
                        step_order TEXT,
                        explain TEXT,
                        PRIMARY KEY (question_id, model_name)
                    )
            ''',
            'human_eval': '''
                CREATE TABLE IF NOT EXISTS whether_gold_answer (
                    question_id TEXT,
                    model_name TEXT,
                    with_gold_answer TEXT,
                    missing_steps TEXT,
                    redundant_steps TEXT,
                    duplicate_steps TEXT,
                    executable TEXT,
                    limitation TEXT,
                    complete TEXT,
                    step_order TEXT,
                    PRIMARY KEY (question_id, model_name, with_gold_answer)
                )
            ''',

        }
        # Create each table
        for table_sql in create_tables_sql.values():
            # print(table_sql)
            cursor.execute(table_sql)

        # Commit the changes and close the connection
        self.conn.commit()

        print("Table Created Successfully")

    def insert_into_question(self, data):
        """
        Insert data into the qa_question table
        :param data: (question_id, template_id, knowledge_id, target_view, question, answer)
        """
        cursor = self.conn.cursor()
        query = '''INSERT INTO question (question_id, category, target_view, limitation, question, answer) VALUES (?, ?, ?, ?, ?, ?)'''
        cursor.execute(query, data)
        self.conn.commit()

    def count_question_rows(self):
        cursor = self.conn.cursor()
        # 使用 SQL 查询语句获取表格中的数据行数
        cursor.execute(f"SELECT COUNT(*) FROM question")
        row_count = cursor.fetchone()[0]
        return row_count

    def clean_table(self, table_name):
        """
            Clear all data of the table based on table_name
        """
        cursor = self.conn.cursor()
        query = f'DELETE FROM {table_name}'
        cursor.execute(query)
        self.conn.commit()

    def insert_into_llm_inference(self, data):
        """
        Insert data into the llm_inference table
        :param data: (question_id, model_name, inference)
        """
        cursor = self.conn.cursor()
        query = '''
        INSERT INTO llm_inference (question_id, model_name, inference)
        VALUES (?, ?, ?)
        ON CONFLICT(question_id, model_name)
        DO UPDATE SET
        inference=excluded.inference
        '''
        cursor.execute(query, data)
        self.conn.commit()

    def insert_into_gold_answer(self, model_name, question_id, gold_answer):
        """
        Insert data into the eval_result table with human_eval as an optional field.
        If human_eval is not provided, it will be stored as NULL in the database.
        """
        cursor = self.conn.cursor()
        query = '''INSERT INTO gold_answer (model_name, question_id, answer) VALUES (?, ?, ?)'''
        try:
            cursor.execute(query, (model_name, question_id, gold_answer))
        except sqlite3.IntegrityError:
            print(f"repetitive primary key {model_name, question_id}")
        self.conn.commit()

    def insert_into_gold_answer_without_learn(self, model_name, question_id, gold_answer):
        """
        Insert data into the eval_result table with human_eval as an optional field.
        If human_eval is not provided, it will be stored as NULL in the database.
        """
        cursor = self.conn.cursor()
        query = '''INSERT INTO gold_answer_without_learn (model_name, question_id, answer) VALUES (?, ?, ?)'''
        try:
            cursor.execute(query, (model_name, question_id, gold_answer))
        except sqlite3.IntegrityError:
            print(f"repetitive primary key {model_name, question_id}")
        self.conn.commit()

    def select_question_ids_from_llm_inference(self, category, model_name, question_table):
        if category == 'ALL':
            cursor = self.conn.cursor()
            query = f'''
                    SELECT li.question_id
                    FROM llm_inference li
                    WHERE li.model_name = ?
            '''
            cursor.execute(query, (category, model_name))
        else:
            cursor = self.conn.cursor()
            query = f'''
            SELECT li.question_id
            FROM llm_inference li JOIN {question_table} qt
            ON li.question_id = qt.question_id
            WHERE qt.category = ? AND li.model_name = ?
            '''
            cursor.execute(query, (category, model_name))
        return cursor.fetchall()

    def select_answer_from_gold_answer(self):
        cursor = self.conn.cursor()
        query = f'''
        SELECT question_id, answer
        FROM gold_answer
        '''
        cursor.execute(query, ())
        return cursor.fetchall()

    def select_gold_answer_without_learn(self):
        cursor = self.conn.cursor()
        query = f'''
            SELECT question_id, answer
            FROM gold_answer_without_learn
        '''
        cursor.execute(query, ())
        return cursor.fetchall()

    def select_answer_from_interfere(self):
        cursor = self.conn.cursor()
        query = f'''
            SELECT question_id, model_name, interfere_category, inference
            FROM interfere
        '''
        cursor.execute(query, ())
        return cursor.fetchall()

    def select_inference_from_llm_inference(self):
        cursor = self.conn.cursor()
        query = f'''
        SELECT question_id, model_name, inference
        FROM llm_inference
        '''
        cursor.execute(query, ())
        return cursor.fetchall()

    def select_gold_answer(self):
        cursor = self.conn.cursor()
        query = f'''
            SELECT model_name, question_id, answer
            FROM gold_answer
        '''
        cursor.execute(query, ())
        return cursor.fetchall()

    def update_answer_from_gold_answer(self, answer, question_id):
        cursor = self.conn.cursor()
        query = f'''
            UPDATE  gold_answer SET answer = ?
            WHERE question_id = ? 
        '''
        cursor.execute(query, (answer, question_id))
        self.conn.commit()

    def update_gold_answer_without_learn(self, answer, question_id):
        cursor = self.conn.cursor()
        query = f'''
            UPDATE  gold_answer_without_learn SET answer = ?
            WHERE question_id = ? 
        '''
        cursor.execute(query, (answer, question_id))
        self.conn.commit()

    def update_interfere(self, interfere, question_id, model_name, interfere_category):
        cursor = self.conn.cursor()
        query = f'''
                    UPDATE  interfere SET inference = ?
                    WHERE question_id = ? AND model_name = ? AND interfere_category = ?
                '''
        cursor.execute(query, (interfere, question_id, model_name, interfere_category))
        self.conn.commit()

    def delete_from_gold_answer(self, model_name, question_id):
        #
        cursor = self.conn.cursor()
        query = f'''
            DELETE FROM gold_answer WHERE model_name LIKE ? AND question_id LIKE ?
        '''
        cursor.execute(query, (model_name, question_id))
        self.conn.commit()

    def update_inference_from_llm_inference(self, question_id, model_name, inference):
        cursor = self.conn.cursor()
        query = f'''
            UPDATE  llm_inference SET inference = ?
            WHERE question_id = ? AND model_name = ?
        '''
        cursor.execute(query, (inference, question_id, model_name))
        self.conn.commit()

    def count_current_gold_answer(self, category):
        cursor = self.conn.cursor()
        query = f'''
            SELECT COUNT(distinct question_id)
            FROM gold_answer
            WHERE model_name = 'gpt-3.5-turbo' AND EXISTS(
                SELECT 1
                FROM question
                WHERE question.question_id = gold_answer.question_id AND category = ?
            )
        '''
        cursor.execute(query, (category,))
        return cursor.fetchone()[0]

    def get_annotated_question_ids(self, category):
        query = """
            SELECT question_id
            FROM abstract_question aq
            WHERE EXISTS(
                SELECT 1
                FROM human_eval he
                WHERE he.question_id = aq.question_id
            )
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (category, ))
        abs_question_ids = cursor.fetchall()
        abs_question_ids = [abs[0] for abs in abs_question_ids]

        question_id_list = []
        for abs_id in abs_question_ids:
            query = """
                SELECT question_id
                FROM question
                WHERE abstract_question_id = ?
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (abs_id,))
            question_ids = cursor.fetchall()
            question_id = random.sample(question_ids, 1)
            question_id_list.append(question_id[0][0])

        return question_id_list



    def get_gpt_3_question_ids(self):
        """
        Get a list of question IDs for GPT-3 from the eval_result table.
        """
        query = """
                SELECT q.question_id, q.category
                FROM llm_inference li
                JOIN question q ON li.question_id = q.question_id
                WHERE EXISTS(
                    SELECT 1
                    FROM gold_answer ga
                    WHERE ga.question_id = li.question_id AND ga.model_name = 'gpt-3.5-turbo'
                )
                GROUP BY q.question_id, q.category
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        question_ids_per_view = {}
        for row in cursor.fetchall():
            question_id, category = row
            if category not in question_ids_per_view:
                question_ids_per_view[category] = []
            question_ids_per_view[category].append(question_id)
        return question_ids_per_view

    def get_question_ids_per_view(self):
        """
        Get a list of question IDs for each target view from the eval_result table.
        """
        query = """
                SELECT q.question_id, q.category
                FROM llm_inference li
                JOIN question q ON li.question_id = q.question_id
                GROUP BY q.question_id, q.category
                """
        cursor = self.conn.cursor()
        cursor.execute(query)
        question_ids_per_view = {}
        for row in cursor.fetchall():
            question_id, category = row
            if category not in question_ids_per_view:
                question_ids_per_view[category] = []
            question_ids_per_view[category].append(question_id)
        return question_ids_per_view

    def insert_into_eval_result(self, eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, explain):
        cursor = self.conn.cursor()
        query = '''INSERT INTO eval_result (eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, explain_1) VALUES (?, ?, ?, ?, ?, ?, ?)'''
        cursor.execute(query, (eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, explain))
        self.conn.commit()

    def update_interfere_eval_result(self, question_id, eval_model_name, missing_steps, redundant_steps, duplicate_steps, explain):
        cursor = self.conn.cursor()
        query = '''UPDATE interfere SET missing_steps = ?, redundant_steps = ?, duplicate_steps = ?, explain_1 = ? WHERE question_id = ? AND model_name = ?'''
        cursor.execute(query, (missing_steps, redundant_steps, duplicate_steps, explain, question_id, eval_model_name))
        self.conn.commit()

    def add_result_in_eval_result(self, eval_model_name, question_id, model_name, limitation, complete, step_order, explain):
        cursor = self.conn.cursor()
        query = '''UPDATE eval_result SET limitation = ?, complete = ?, step_order = ?, explain_2 = ? WHERE eval_model_name = ? AND question_id= ? AND model_name = ?'''
        cursor.execute(query, (limitation, complete, step_order, explain, eval_model_name, question_id, model_name))
        self.conn.commit()

    def add_result_in_interfere(self, eval_model_name, question_id, model_name, limitation, complete, step_order, explain):
        cursor = self.conn.cursor()
        query = '''UPDATE interfere SET limitation = ?, complete = ?, step_order = ?, explain_2 = ? WHERE question_id= ? AND model_name = ?'''
        cursor.execute(query, (limitation, complete, step_order, explain, question_id, model_name))
        self.conn.commit()

    def drop_table(self, table_name):
        """
        Drop a table from the database.
        """
        cursor = self.conn.cursor()
        query = f"DROP TABLE IF EXISTS {table_name}"
        cursor.execute(query)
        self.conn.commit()

    def select_one_answer_from_gold_answer(self, question_id, model_name):
        query = """
                SELECT question_id
                FROM gold_answer
                WHERE question_id = ? and model_name = ?
                """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id, model_name))
        result = cursor.fetchall()
        return result

    def select_one_answer_from_gold_answer_without_learn(self, question_id, model_name):
        query = """
                SELECT question_id
                FROM gold_answer_without_learn
                WHERE question_id = ? and model_name = ?
                """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id, model_name))
        result = cursor.fetchall()
        return result

    def select_eval_result_ids(self, eval_model):
        query = """
            SELECT eval_model_name, question_id, model_name
            FROM eval_result
            WHERE limitation is NULL AND eval_model_name = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (eval_model,))
        result = cursor.fetchall()
        return result

    def update_eval_result(self, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, explain):
        cursor = self.conn.cursor()
        query = '''UPDATE eval_result SET question_id = ?, model_name = ? WHERE missing_steps = ? AND redundant_steps = ? AND duplicate_steps = ? AND explain_1 = ?'''
        cursor.execute(query, (question_id, model_name, missing_steps, redundant_steps, duplicate_steps, explain))
        self.conn.commit()

    def select_inference_of_01ai(self):
        cursor = self.conn.cursor()
        query = f'''
            SELECT *
            FROM llm_inference
            WHERE model_name like '01ai%'
        '''
        cursor.execute(query, ())
        return cursor.fetchall()

    def clean_inference_of_01ai(self):
        cursor = self.conn.cursor()
        query = f'''
            DELETE FROM llm_inference
            WHERE model_name like '01ai%'
        '''
        cursor.execute(query, ())
        self.conn.commit()

    def update_answer_length(self):
        cursor = self.conn.cursor()
        query = f'''
            SELECT *
            FROM gold_answer
        '''
        cursor.execute(query, ())
        answer = cursor.fetchall()

        for answer in tqdm(answer):
            question_id = answer[1]
            steps_list = re.findall(r'\d+\.\s.*?(?=\d+\.\s|\Z)', answer[2], re.DOTALL)

            answer_length = len(steps_list)

            query = f'''
                UPDATE question
                SET answer_length = {answer_length}
                WHERE question_id = {question_id}
            '''
            cursor.execute(query, ())
            self.conn.commit()
            # exit(0)

    def get_gold_answer_question_ids(self, eval_model):
        cursor = self.conn.cursor()
        query = f'''
            SELECT question_id
            FROM gold_answer WHERE model_name like '{eval_model}'
        '''
        cursor.execute(query, ())
        answer = cursor.fetchall()
        answer = [int(item[0]) for item in answer]
        return answer

    def clean_question_less_limitation(self):
        cursor = self.conn.cursor()
        query = f'''
            SELECT abstract_question_id, count(*)
            FROM question
            GROUP BY abstract_question_id
        '''
        cursor.execute(query, ())
        answer = cursor.fetchall()

        for one_answer in answer:
            abstract_question_id = one_answer[0]
            question_num = one_answer[1]
            if question_num < 3:
                query = f'''
                    DELETE FROM question
                    WHERE abstract_question_id = {abstract_question_id}
                '''
                cursor.execute(query, ())
                self.conn.commit()

    def get_executable_data(self):
        cursor = self.conn.cursor()
        query = "SELECT eval_model_name, question_id, model_name, executable FROM eval_result"
        cursor.execute(query, ())
        answer = cursor.fetchall()
        return answer

    def update_executable_data(self, eval_model, question_id, model_name, executable):
        cursor = self.conn.cursor()
        query = '''
            UPDATE eval_result
            SET executable = ?
            WHERE eval_model_name = ? AND question_id = ? AND model_name = ?
        '''
        cursor.execute(query, (executable, eval_model, question_id, model_name))
        self.conn.commit()


def replace_executable_data():
    repair_data = EvalDataBase("./data/database/repair.db")
    script_data = EvalDataBase("./data/database/script.db")

    executable_data = repair_data.get_executable_data()
    for data in tqdm(executable_data, desc='update executable'):
        eval_model, question_id, model_name, executable = data
        script_data.update_executable_data(eval_model, question_id, model_name, executable)





if __name__ == "__main__":
    db = EvalDataBase("data/database/script.db")
    db.table_create()
    # replace_executable_data()

    # db.update_answer_length()
    # db.clean_inference_of_01ai()
    # db.drop_table('question')
    # db.clean_table('gold_answer')
    # db.clean_table('eval_result')
