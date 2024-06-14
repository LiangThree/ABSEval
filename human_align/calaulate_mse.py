import numpy as np
import sqlite3

class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def get_human_eval_question(self, category):
        result_list = []
        for one_category in category:
            query = """
                SELECT human_eval.* FROM human_eval JOIN question ON human_eval.question_id = question.question_id
                WHERE question.category = ?
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (one_category,))
            result = cursor.fetchall()
            result_list.extend(result[:15])
        return result_list


    def get_gpt_3_eval(self, model_name, question_id):
        query = """
                        SELECT * FROM eval_result
                        WHERE eval_model_name = 'gpt-3.5-turbo' AND model_name = ? AND question_id = ?
                    """
        cursor = self.conn.cursor()
        cursor.execute(query, (model_name, question_id))
        result = cursor.fetchone()
        return result

    def get_gpt_4_eval(self, model_name, question_id):
        query = """
            SELECT * FROM eval_result
            WHERE eval_model_name = 'gpt-4-turbo' AND model_name = ? AND question_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (model_name, question_id,))
        result = cursor.fetchone()
        return result

    def get_qwen_eval(self, model_name, question_id):
        query = """
            SELECT * FROM eval_result
            WHERE eval_model_name = 'Qwen-110B-Chat' AND model_name = ? AND question_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (model_name, question_id,))
        result = cursor.fetchone()
        return result


eval_db = EvalDataBase('data/database/script.db')

def calculate_mse(true_values, predicted_values):
    # 确保输入是numpy数组
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    # 计算MSE
    mse = np.mean((true_values - predicted_values) ** 2)
    mse = round(float(mse), 3)

    return mse


def get_data():
    category = ["Arts and Entertainment", "Computers and Electronics", "Education and Communications",
                "Food and Entertaining", "Health", "Hobbies and Crafts", "Holidays and Traditions",
                "Home and Garden", "Sports and Fitness", "Travel"]
    eval_result = eval_db.get_human_eval_question(category)


    human_eval_list = []
    gpt_3_eval_list = []
    gpt_4_eval_list = []
    qwen_eval_list = []

    for one_eval_result in eval_result:
        question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order = one_eval_result

        current_gpt_3_eval = eval_db.get_gpt_3_eval(model_name, question_id)
        if current_gpt_3_eval is None:
            continue
        else:
            current_gpt_3_eval = current_gpt_3_eval[3:10]
        current_gpt_3_eval = [1 if x == 'True' else 0 for x in current_gpt_3_eval]

        current_gpt_4_eval = eval_db.get_gpt_4_eval(model_name, question_id)[3:10]
        current_gpt_4_eval = [1 if x == 'True' else 0 for x in current_gpt_4_eval]

        current_qwen_eval = eval_db.get_qwen_eval(model_name, question_id)[3:10]
        current_qwen_eval = [1 if x == 'True' else 0 for x in current_qwen_eval]

        if current_gpt_3_eval is not None and current_gpt_4_eval  is not None and current_qwen_eval  is not None:
            human_eval_list.append([1 if x == 'True' else 0 for x in
                                    [missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete,
                                     step_order]])
            gpt_3_eval_list.append(current_gpt_3_eval)
            gpt_4_eval_list.append(current_gpt_4_eval)
            qwen_eval_list.append(current_qwen_eval)

    gpt_3_mse = calculate_mse(human_eval_list, gpt_3_eval_list)
    gpt_4_mse = calculate_mse(human_eval_list, gpt_4_eval_list)
    qwen_mse = calculate_mse(human_eval_list, qwen_eval_list)

    # print(gpt_3_mse, gpt_4_mse, qwen_mse)
    print('|Eval Model|MSE|')
    print('|:--|:--|')
    print(f'|GPT-3.5-turbo|{gpt_3_mse}|')
    print(f'|GPT-4-turbo|{gpt_4_mse}|')
    print(f'|Qwen-110B|{qwen_mse}|')



if __name__ == '__main__':
    get_data()
