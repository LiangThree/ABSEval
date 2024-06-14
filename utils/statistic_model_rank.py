import sqlite3
from tqdm import tqdm


class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def get_eval_model(self):
        query = """
            SELECT distinct model_name FROM model_rank
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        result = [one[0] for one in result]
        return result

    def get_model_score(self, model_name):
        query = """
            SELECT MASEval, Qwen, Rouge, BERTScore FROM model_rank WHERE model_name = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (model_name,))
        result = cursor.fetchall()
        # result = [one[0] for one in result]
        return result

    def update_MASEval(self):
        query = f"""
            SELECT question_id, model_name
            FROM model_rank
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()

        for one_result in tqdm(result):
            question_id = one_result[0]
            model_name = one_result[1]
            query = f"""
                SELECT missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order
                FROM eval_result
                WHERE question_id = ? AND model_name = ?
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (question_id, model_name))
            result = cursor.fetchone()
            sum = 0
            if result[0] == 'False':
                sum += 1
            if result[1] == 'False':
                sum += 1
            if result[2] == 'False':
                sum += 1
            if result[4] == 'True':
                sum += 1
            if result[5] == 'True':
                sum += 1
            if result[6] == 'True':
                sum += 1
            query = f"""
                UPDATE model_rank
                SET MASEval = ?
                WHERE question_id = ? AND model_name = ?
            """
            cursor = self.conn.cursor()
            cursor.execute(query, (sum, question_id, model_name))
            self.conn.commit()
        # return result


def calculate_model_score(model_list):
    model_score_dict = {}
    for model_name in model_list:
        maseval_sum = 0
        qwen_sum = 0
        rouge_sum = 0
        bertscore_sum = 0
        eval_list = eval_db.get_model_score(model_name)
        for one_score in eval_list:
            maseval, qwen, rouge, bertscore = one_score
            if qwen is None:
                qwen = 0
            maseval_sum += float(maseval)
            qwen_sum += float(qwen)
            rouge_sum += float(rouge)
            bertscore_sum += float(bertscore)
        maseval_sum = round(maseval_sum / len(eval_list), 2)
        qwen_sum = round(qwen_sum / len(eval_list), 2)
        rouge_sum = round(rouge_sum / len(eval_list), 2)
        bertscore_sum = round(bertscore_sum / len(eval_list), 2)

        model_score_dict[model_name] = {}
        model_score_dict[model_name]['MASEval'] = maseval_sum
        model_score_dict[model_name]['Qwen'] = qwen_sum
        model_score_dict[model_name]['Rouge'] = rouge_sum
        model_score_dict[model_name]['BERTScore'] = bertscore_sum

    return model_score_dict


def sort_models_by_scores(model_dict):
    # 将模型按评估指标分数进行排序
    sorted_by_maseval = sorted(model_dict.items(), key=lambda x: x[1]['MASEval'], reverse=True)
    sorted_by_qwen = sorted(model_dict.items(), key=lambda x: x[1]['Qwen'], reverse=True)
    sorted_by_rouge = sorted(model_dict.items(), key=lambda x: x[1]['Rouge'], reverse=True)
    sorted_by_bertscore = sorted(model_dict.items(), key=lambda x: x[1]['BERTScore'], reverse=True)

    return {
        'MASEval': [one[0] for one in sorted_by_maseval],
        'Qwen': [one[0] for one in sorted_by_qwen],
        'Rouge': [one[0] for one in sorted_by_rouge],
        'BERTScore': [one[0] for one in sorted_by_bertscore]
    }

eval_db = EvalDataBase('data/database/script.db')


if __name__ == '__main__':
    # eval_db.update_MASEval()
    model_list = eval_db.get_eval_model()
    model_score_dict = calculate_model_score(model_list)
    model_rank = sort_models_by_scores(model_score_dict)
    # print(model_rank)
    for model in model_rank['BERTScore']:
        print(model)

    print('----------------------------------------')
    for model in model_rank['Rouge']:
        print(model)
