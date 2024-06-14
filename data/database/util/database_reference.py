import json
import sqlite3
import random
import os
import yaml
from collections import Counter
from tqdm import tqdm


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
            'choices_question': '''
                CREATE TABLE IF NOT EXISTS choices_question (
                    question_id TEXT PRIMARY KEY,
                    template_id TEXT,
                    knowledge_id TEXT,
                    question_type TEXT DEFAULT 'choices_question',
                    target_view TEXT,
                    question TEXT,
                    choices TEXT,
                    gold_tag TEXT
                )
            ''',
            'qa_question': '''
                CREATE TABLE IF NOT EXISTS qa_question (
                    question_id TEXT PRIMARY KEY,
                    template_id TEXT,
                    knowledge_id TEXT,
                    question_type TEXT DEFAULT 'qa_question',
                    target_view TEXT,
                    question TEXT,
                    answer TEXT
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
            'human_eval': '''
                CREATE TABLE IF NOT EXISTS human_eval (
                    question_id TEXT,
                    model_name TEXT,
                    human_eval TEXT,
                    PRIMARY KEY (question_id, model_name)
                )
            ''',
            'eval_result': '''
                CREATE TABLE IF NOT EXISTS eval_result (
                    question_id TEXT,
                    model_name TEXT,
                    eval_model_name TEXT,
                    eval_result TEXT,
                    PRIMARY KEY (question_id, model_name, eval_model_name)
                )
            ''',
            'template': '''
                CREATE TABLE IF NOT EXISTS template (
                    template_id TEXT PRIMARY KEY,
                    template TEXT)
            '''
        }
        # Create each table
        for table_sql in create_tables_sql.values():
            cursor.execute(table_sql)

        # Commit the changes and close the connection
        self.conn.commit()

    def insert_into_choices_question(self, data):
        """
        Insert data into the choices_question table
        :param data: (question_id, template_id, knowledge_id, target_view, question, choices, gold_tag)
        """
        cursor = self.conn.cursor()
        query = '''INSERT INTO choices_question (question_id, template_id, knowledge_id, target_view, question, choices, gold_tag) VALUES (?, ?, ?, ?, ?, ?, ?)'''
        cursor.execute(query, data)
        self.conn.commit()

    def insert_into_qa_question(self, data):
        """
        Insert data into the qa_question table
        :param data: (question_id, template_id, knowledge_id, target_view, question, answer)
        """
        cursor = self.conn.cursor()
        query = '''INSERT INTO qa_question (question_id, template_id, knowledge_id, target_view, question, answer) VALUES (?, ?, ?, ?, ?, ?)'''
        cursor.execute(query, data)
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

    def insert_into_human_eval(self, data):
        """
        Insert data into the human_eval table
        :param data: (question_id, model_name, human_eval)
        """
        cursor = self.conn.cursor()
        print(data)
        query = '''
        INSERT INTO human_eval (question_id, model_name, human_eval) 
        VALUES (?, ?, ?) 
        ON CONFLICT(question_id, model_name) 
        DO UPDATE SET 
        human_eval=excluded.human_eval
        '''
        cursor.execute(query, data)
        self.conn.commit()

    def insert_into_eval_result(self, question_id, model_name, eval_model_name, eval_result):
        """
        Insert data into the eval_result table with human_eval as an optional field.
        If human_eval is not provided, it will be stored as NULL in the database.
        """
        cursor = self.conn.cursor()
        query = '''
        INSERT INTO eval_result (question_id, model_name, eval_model_name, eval_result) 
        VALUES (?, ?, ?, ?)
        ON CONFLICT (question_id, model_name, eval_model_name)
        DO UPDATE SET
        eval_result=excluded.eval_result
        '''
        cursor.execute(query, (question_id, model_name, eval_model_name, eval_result))
        self.conn.commit()

    def insert_into_template(self, data):
        """
        Insert data into the template table
        :param data: (template_id, template)
        """
        cursor = self.conn.cursor()
        query = '''INSERT INTO template (template_id, template) VALUES (?, ?)'''
        cursor.execute(query, data)
        self.conn.commit()

    def delete_from_choices_question(self, question_id):
        """
        Delete a record from the choices_question table based on question_id
        """
        cursor = self.conn.cursor()
        query = '''DELETE FROM choices_question WHERE question_id = ?'''
        cursor.execute(query, (question_id,))
        self.conn.commit()

    def delete_from_qa_question(self, question_id):
        """
        Delete a record from the qa_question table based on question_id
        """
        cursor = self.conn.cursor()
        query = '''DELETE FROM qa_question WHERE question_id = ?'''
        cursor.execute(query, (question_id,))
        self.conn.commit()

    def delete_from_llm_inference(self, question_id, model_name):
        """
        Delete a record from the llm_inference table based on question_id and model_name
        """
        cursor = self.conn.cursor()
        query = '''DELETE FROM llm_inference WHERE question_id = ? AND model_name = ?'''
        cursor.execute(query, (question_id, model_name))
        self.conn.commit()

    def delete_from_llm_inference_by_model(self, model_name):
        """
        Delete all records from the llm_inference table based on model_name.
        :param model_name: The name of the model to delete records for.
        """
        cursor = self.conn.cursor()
        query = '''DELETE FROM llm_inference WHERE model_name = ?'''
        cursor.execute(query, (model_name,))
        self.conn.commit()

    def delete_from_human_eval(self, question_id, model_name):
        """
        Delete a record from the human_eval table based on question_id and model_name
        """
        cursor = self.conn.cursor()
        query = '''DELETE FROM human_eval WHERE question_id = ? AND model_name = ?'''
        cursor.execute(query, (question_id, model_name))
        self.conn.commit()

    def delete_from_eval_result(self, question_id, model_name, eval_model_name):
        """
        Delete a record from the eval_result table based on question_id、model_name and eval_model_name
        """
        cursor = self.conn.cursor()
        query = '''DELETE FROM eval_result WHERE question_id = ? AND model_name = ? AND eval_model_name = ?'''
        cursor.execute(query, (question_id, model_name, eval_model_name))
        self.conn.commit()

    def delete_from_template(self, template_id):
        """
        Delete a record from the template table based on template_id
        """
        cursor = self.conn.cursor()
        query = '''DELETE FROM template WHERE template_id = ?'''
        cursor.execute(query, (template_id,))
        self.conn.commit()

    def clean_table(self, table_name):
        """
            Clear all data of the table based on table_name
        """
        cursor = self.conn.cursor()
        query = f'DELETE FROM {table_name}'
        cursor.execute(query)
        self.conn.commit()

    """
    ***************************
        Annotation Methods
    ***************************
    """
    def sample_one_for_human_annotate(self):
        cursor = self.conn.cursor()

        # 获取所有 qa_question 记录
        cursor.execute('SELECT question_id, question, answer FROM qa_question')
        all_qa_records = cursor.fetchall()

        # 随机打乱记录顺序
        random.shuffle(all_qa_records)

        for qa_record in all_qa_records:
            question_id, question, answer = qa_record

            # 从 llm_inference 中检索相同 question_id 的所有记录
            cursor.execute('''
                SELECT model_name, inference 
                FROM llm_inference 
                WHERE question_id = ?
            ''', (question_id,))
            llm_records = cursor.fetchall()
            result = []

            for model_name, inference in llm_records:
                # 检查这个记录是否在 human_eval 中
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM human_eval 
                    WHERE question_id = ? AND model_name = ?
                ''', (question_id, model_name))

                if cursor.fetchone()[0] == 0:
                    result.append((question_id, question, inference, answer, model_name))

            if result:
                # 将结果转换为字典
                return [{'question_id': q_id, 'question': q, 'inference': inf, 'answer': ans, 'model_name': m_name}
                        for q_id, q, inf, ans, m_name in result][0]  # 返回一个即可

        # 所有记录都被检查过，没有找到符合条件的数据
        return None

    def sample_one_for_human_annotate_by_view(self, target_view):
        cursor = self.conn.cursor()

        # 从 choices_question 表中获取特定 target_view 的所有记录
        cursor.execute('SELECT question_id, question, answer FROM qa_question WHERE target_view = ?',
                       (target_view,))
        all_qa_records = cursor.fetchall()

        # 随机打乱记录顺序
        random.shuffle(all_qa_records)

        for qa_record in all_qa_records:
            question_id, question, answer = qa_record

            # 从 llm_inference 表中检索相同 question_id 的所有记录
            cursor.execute('''
                SELECT model_name, inference 
                FROM llm_inference 
                WHERE question_id = ?
            ''', (question_id,))
            llm_records = cursor.fetchall()

            result = []

            for model_name, inference in llm_records:
                # 检查这个记录是否已经在 human_eval 表中
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM human_eval 
                    WHERE question_id = ? AND model_name = ?
                ''', (question_id, model_name))

                if cursor.fetchone()[0] == 0:
                    result.append((question_id, question, inference, answer, model_name))

            if result:
                # 将结果转换为字典
                res = [{'question_id': q_id, 'question': q, 'inference': inf, 'answer': ans, 'model_name': m_name}
                       for q_id, q, inf, ans, m_name in result]
                return random.choice(res)  # 返回一个即可

        # 如果没有找到符合条件的记录，则返回 None
        return None

    def get_target_views_in_qa(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT DISTINCT target_view FROM qa_question')
        return [row[0] for row in cursor.fetchall()]

    @staticmethod
    def get_database_list(db_directory="database/"):
        """
        Get a list of all database files in a specific directory.
        """
        database_files = [file for file in os.listdir(db_directory) if file.endswith('.db')]
        return database_files

    def get_total_questions_per_view(self):
        cursor = self.conn.cursor()
        query = '''
            SELECT q.target_view, COUNT(*) as total_count 
            FROM llm_inference l
            JOIN qa_question q ON l.question_id = q.question_id
            GROUP BY q.target_view
        '''
        cursor.execute(query)
        return cursor.fetchall()

    def get_annotated_questions_per_view(self):
        cursor = self.conn.cursor()
        query = '''
            SELECT target_view, COUNT(*) as annotated_count 
            FROM (
                SELECT DISTINCT q.target_view, h.question_id, h.model_name
                FROM human_eval h
                JOIN llm_inference l ON h.question_id = l.question_id AND h.model_name = l.model_name
                JOIN qa_question q ON l.question_id = q.question_id
            ) AS unique_annotations
            GROUP BY target_view
        '''
        cursor.execute(query)
        return cursor.fetchall()

    def get_sample_data(self, question_id, model_name):
        """
        Retrieve sample data based on question_id and model_name.
        """
        cursor = self.conn.cursor()

        # 查询 choices_question 表中的数据
        question_query = '''
            SELECT question_id, question, answer
            FROM qa_question
            WHERE question_id = ?
        '''
        cursor.execute(question_query, (question_id,))
        question_data = cursor.fetchone()

        # 查询 llm_inference 表中的数据
        inference_query = '''
            SELECT inference
            FROM llm_inference
            WHERE question_id = ? AND model_name = ?
        '''
        cursor.execute(inference_query, (question_id, model_name))
        inference_data = cursor.fetchone()

        if question_data and inference_data:
            # 构造并返回一个包含所有相关信息的字典
            return {
                'question_id': question_data[0],
                'question': question_data[1],
                'answer': question_data[2],
                'inference': inference_data[0],
                'model_name': model_name
            }
        else:
            return None  # 如果没有找到数据，则返回 None

    def get_inference_count_per_view_and_model(self):
        cursor = self.conn.cursor()
        query = '''
            SELECT q.target_view, l.model_name, COUNT(*) as inference_count 
            FROM llm_inference l
            JOIN qa_question q ON l.question_id = q.question_id
            GROUP BY q.target_view, l.model_name
        '''
        cursor.execute(query)
        return cursor.fetchall()

    # 这个函数应该返回一个包含三个字段的列表：target_view、model_name 和已标注数量（annotated_count）
    def get_annotated_count_by_model(self):
        cursor = self.conn.cursor()
        query = '''
            SELECT q.target_view, l.model_name, COUNT(*) as annotated_count
            FROM human_eval h
            JOIN llm_inference l ON h.question_id = l.question_id AND h.model_name = l.model_name
            JOIN qa_question q ON l.question_id = q.question_id
            WHERE h.human_eval IS NOT NULL
            GROUP BY q.target_view, l.model_name
        '''
        cursor.execute(query)
        return cursor.fetchall()

    """
        ***************************
            llm-eval-tool Methods
        ***************************
    """
    def select_inference_from_llm_inference(self, question_id, model_name):
        cursor = self.conn.cursor()
        query = '''SELECT inference FROM llm_inference WHERE question_id = ? AND model_name = ?'''
        cursor.execute(query, (question_id, model_name))
        return cursor.fetchone()

    def select_question_ids_from_choices_question(self, target_view):
        cursor = self.conn.cursor()
        query = '''SELECT question_id FROM choices_question WHERE target_view = ?'''
        cursor.execute(query, (target_view,))
        return cursor.fetchall()

    def select_from_choices_question_by_question_ids(self, question_ids):
        cursor = self.conn.cursor()
        query = '''SELECT * FROM choices_question WHERE question_id IN ({})'''.format(','.join('?' * len(question_ids)))
        cursor.execute(query, question_ids)
        return cursor.fetchall()

    def select_question_ids_from_llm_inference(self, target_view, model_name, question_table):
        cursor = self.conn.cursor()
        query = f'''
        SELECT li.question_id
        FROM llm_inference li JOIN {question_table} qt
        ON li.question_id = qt.question_id
        WHERE qt.target_view = ? AND li.model_name = ?
        '''
        cursor.execute(query, (target_view, model_name))
        return cursor.fetchall()

    def select_from_qa_question_by_question_ids(self, question_ids):
        cursor = self.conn.cursor()
        query = '''SELECT * FROM qa_question WHERE question_id IN ({})'''.format(','.join('?' * len(question_ids)))
        cursor.execute(query, question_ids)
        return cursor.fetchall()

    """
        ***************************
            Web-platform Methods
        ***************************
    """
    def organize_eval_results_for_qa_questions(self):
        cursor = self.conn.cursor()
        # 获取 eval_result 表中所有条目
        cursor.execute("SELECT question_id, model_name, eval_model_name, eval_result FROM eval_result")
        eval_results = cursor.fetchall()

        # 获取 qa_question 表中所有问题的 target_view
        cursor.execute("SELECT question_id, target_view FROM qa_question")
        qa_questions = {qid: tv for qid, tv in cursor.fetchall()}

        # 初始化 JSON 结构
        organized_results = {"qa_question": {}}

        for question_id, model_name, eval_model_name, result in tqdm(eval_results, desc="Organizing eval_results for qa_questions"):
            if question_id in qa_questions:
                target_view = qa_questions[question_id]

                # 按层级组织数据
                if target_view not in organized_results["qa_question"]:
                    organized_results["qa_question"][target_view] = {}
                if eval_model_name not in organized_results["qa_question"][target_view]:
                    organized_results["qa_question"][target_view][eval_model_name] = {}
                if model_name not in organized_results["qa_question"][target_view][eval_model_name]:
                    organized_results["qa_question"][target_view][eval_model_name][model_name] = []

                # 添加 question_id 和 eval_result
                organized_results["qa_question"][target_view][eval_model_name][model_name].append({
                    "question_id": question_id,
                    "eval_result": result
                })

        return organized_results

    def organize_and_calculate_accuracy_for_qa_questions(self):
        # 首先组织数据
        organized_results = self.organize_eval_results_for_qa_questions()

        # 然后计算每个模型的正确率
        # 计算总迭代次数
        total_iterations = 0
        for target_view in organized_results["qa_question"]:
            for eval_model_name in organized_results["qa_question"][target_view]:
                for model_name in organized_results["qa_question"][target_view][eval_model_name]:
                    total_iterations += len(organized_results["qa_question"][target_view][eval_model_name][model_name])

        # 创建进度条
        progress_bar = tqdm(total=total_iterations, desc="Calculating accuracy for each model in qa_question")

        for target_view in organized_results["qa_question"]:
            for eval_model_name in organized_results["qa_question"][target_view]:
                for model_name in organized_results["qa_question"][target_view][eval_model_name]:
                    # 获取特定模型的所有评估结果
                    evaluations = organized_results["qa_question"][target_view][eval_model_name][model_name]

                    # 统计正确答案的数量
                    correct_count = 0
                    for _eval in evaluations:
                        eval_result = json.loads(_eval["eval_result"])
                        # 检查 answer 是否为 "1" 或 1
                        if eval_result.get("answer") in ["1", 1]:
                            correct_count += 1
                        progress_bar.update(1)

                    # 计算正确率
                    accuracy = round(correct_count / len(evaluations), 4) if evaluations else 0

                    # 将正确率添加到 JSON 结构中
                    organized_results["qa_question"][target_view][eval_model_name][model_name] = {
                        "accuracy": accuracy,
                        "evaluations": evaluations
                    }
        progress_bar.close()
        return organized_results

    def load_eval_models_for_voting(self, knowledge_capacity):
        """
        Load evaluation models for voting from a YAML file.
        """
        with open("vote_model_config.yaml", 'r') as file:
            eval_models = yaml.safe_load(file)
        capacity_eval_models = eval_models[knowledge_capacity]["eval_vote_model"]
        if len(capacity_eval_models) != 5:
            raise ValueError("Please set 5 eval models for vote")
        return capacity_eval_models

    def get_vote_result(self, question_id, inference_model, eval_models):
        counter = Counter()
        for eval_model in eval_models:
            model_result = self.get_model_eval(question_id, inference_model, eval_model)
            if model_result is None:
                continue
            model_result = float(json.loads(model_result)['answer'])
            counter[model_result] += 1
        if counter.most_common(1):
            return counter.most_common(1)[0][0]
        else:
            return None

    def get_model_eval(self, question_id, inference_model, eval_model):
        query = """
        SELECT eval_result FROM eval_result WHERE question_id=? AND model_name=? AND eval_model_name=?
        """
        cursor = self.conn.cursor()
        row = cursor.execute(query, (question_id, inference_model, eval_model)).fetchone()
        if row is None:
            return None
        else:
            return row[0]

    def get_question_ids_per_view(self):
        """
        Get a list of question IDs for each target view from the eval_result table.
        """
        query = """
        SELECT q.question_id, q.target_view
        FROM eval_result er
        JOIN qa_question q ON er.question_id = q.question_id
        GROUP BY q.question_id, q.target_view
        """
        cursor = self.conn.cursor()
        cursor.execute(query)
        question_ids_per_view = {}
        for row in cursor.fetchall():
            question_id, target_view = row
            if target_view not in question_ids_per_view:
                question_ids_per_view[target_view] = []
            question_ids_per_view[target_view].append(question_id)
        return question_ids_per_view

    def calculate_vote_accuracy(self, eval_models):
        vote_accuracy_results = {}
        question_ids_per_view = self.get_question_ids_per_view()
        total_iterations = sum(
            len(self.get_evaluated_model_names(tv)) * len(q_ids) for tv, q_ids in question_ids_per_view.items())
        progress_bar = tqdm(total=total_iterations, desc="Calculating vote accuracy")

        # 遍历每个 target_view 和 question_id
        for target_view, question_ids in question_ids_per_view.items():
            vote_accuracy_results[target_view] = {}

            # 获取每个 target_view 下被评价的 model_name 列表
            model_names = self.get_evaluated_model_names(target_view)

            for model_name in model_names:
                vote_accuracy_results[target_view][model_name] = {}
                vote_accuracy_results[target_view][model_name]["accuracy"] = self.calculate_model_vote_accuracy(question_ids, model_name, eval_models, progress_bar)
        progress_bar.close()
        return vote_accuracy_results

    def get_evaluated_model_names(self, target_view):
        query = """
        SELECT DISTINCT er.model_name
        FROM eval_result er
        JOIN qa_question q ON er.question_id = q.question_id
        WHERE q.target_view = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (target_view,))
        return [row[0] for row in cursor.fetchall()]

    def calculate_model_vote_accuracy(self, question_ids, model_name, eval_models, progress_bar):
        vote_count = 0
        correct_votes = 0
        for question_id in question_ids:
            vote_result = self.get_vote_result(question_id, model_name, eval_models)
            if vote_result is not None:
                vote_count += 1
                if vote_result in ["1", 1]:
                    correct_votes += 1
            progress_bar.update(1)  # 更新进度条
        if vote_count > 0:
            return round(correct_votes / vote_count, 4)
        else:
            return 0

    def get_target_views_in_cq(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT DISTINCT target_view FROM choices_question')
        return [row[0] for row in cursor.fetchall()]

    def get_models_in_cq(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT model_name FROM llm_inference")
        return [row[0] for row in cursor.fetchall()]

    def calculate_cq_by_target_view_and_model(self):
        cursor = self.conn.cursor()
        target_views = self.get_target_views_in_cq()
        models = self.get_models_in_cq()
        # 计算总迭代次数
        total_iterations = len(target_views) * len(models)
        progress_bar = tqdm(total=total_iterations, desc="Calculating choice question accuracy")

        choice_accuracy_results = {}
        for target_view in target_views:
            choice_accuracy_results[target_view] = {}
            choice_accuracy_results[target_view]["final_result"] = {}
            for model in models:
                query = f"""
                SELECT li.inference, cq.gold_tag 
                FROM llm_inference li
                JOIN choices_question cq ON li.question_id = cq.question_id 
                WHERE li.model_name=? AND cq.target_view=?"""
                rows = cursor.execute(query, (model, target_view,)).fetchall()
                if rows:
                    counter = Counter()
                    for row in rows:
                        if row[0] == row[1]:
                            counter['true'] += 1
                        else:
                            counter['false'] += 1
                    acc = round(counter['true'] / (counter['true'] + counter['false']), 4)
                    choice_accuracy_results[target_view]["final_result"][model] = {"accuracy": acc}
                # 更新进度条
                progress_bar.update(1)
        # 关闭进度条
        progress_bar.close()
        return choice_accuracy_results

    # 修改后的 save_qa_evaluation_results 方法
    def save_qa_evaluation_results(self, out_path, knowledge_capacity):
        eval_models = self.load_eval_models_for_voting(knowledge_capacity)

        organized_results = self.organize_and_calculate_accuracy_for_qa_questions()

        # 计算投票准确率
        vote_accuracy_results = self.calculate_vote_accuracy(eval_models)

        # 将投票准确率添加到 JSON 结构中
        for target_view in vote_accuracy_results:
            organized_results["qa_question"][target_view]["final_result"] = vote_accuracy_results[target_view]

        # 按照target_view和model_name计算选择题准确率
        choice_accuracy_results = self.calculate_cq_by_target_view_and_model()
        print("******** Processing data for saving ********")
        organized_results["choice_question"] = {}
        # 将投票准确率添加到 JSON 结构中
        for target_view in choice_accuracy_results:
            organized_results["choice_question"][target_view] = choice_accuracy_results[target_view]

        # 移除 evaluations 详细数据
        for target_view in organized_results["qa_question"]:
            for eval_model_name in organized_results["qa_question"][target_view]:
                if eval_model_name != "final_result":
                    for model_name in organized_results["qa_question"][target_view][eval_model_name]:
                        del organized_results["qa_question"][target_view][eval_model_name][model_name]["evaluations"]

        # 将结果保存到指定的文件
        with open(out_path, 'w') as out_file:
            json.dump(organized_results, out_file, indent=4)


if __name__ == "__main__":
    db = EvalDataBase("/netcache/liangsirui/llm-eval-tool/data/database/ATOMIC.db")
    db.save_qa_evaluation_results("commonsense.json", 'common')
