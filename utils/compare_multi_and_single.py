import  sqlite3
from proplot import rc
import matplotlib.pyplot as plt
from tqdm import *


def plot_tuples(tuple1, tuple2, x_labels=None):
    plt.rcParams.update({'font.size': 30})
    rc["font.family"] = "Times New Roman"
    # 确保两个元组长度相同
    if len(tuple1) != len(tuple2):
        raise ValueError("Both tuples must have the same length")

    x = range(len(tuple1))  # x轴上的点，0到n-1
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(x, tuple1, marker='o', label='Single Agent', linestyle='-', color='#FA7070FF', markerfacecolor='white',
             markeredgewidth=3, markersize=20, linewidth=3)
    plt.plot(x, tuple2, marker='^', label='ABSEval', linestyle='-', color='#7AA2E3FF', markerfacecolor='white',
             markeredgewidth=3, markersize=20, linewidth=3)

    if x_labels is not None:
        plt.xticks(ticks=x, labels=x_labels, rotation=30)
    else:
        plt.xticks(ticks=x, rotation=30)  # 使用默认的索引作为标签

    plt.ylim(0.4, 1)

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Acc.')
    # plt.title('Line Plot of Two Tuples')
    plt.legend()
    plt.grid(True)
    plt.savefig('utils/graph/line_plot.svg')


class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def get_model_eval(self):
        query = """
            SELECT * FROM model_eval WHERE EXISTS(SELECT 1 FROM human_eval WHERE model_eval.model_name = human_eval.model_name AND model_eval.question_id = human_eval.question_id)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        result = [[one[0], one[1], one[3], one[4], one[5], one[6], one[7], one[8], one[9]] for one in result]
        return result

    def get_MASEval(self):
        query = """
            SELECT * FROM eval_result WHERE eval_model_name = 'Qwen-110B-Chat' AND EXISTS(SELECT 1 FROM human_eval WHERE human_eval.model_name = eval_result.model_name AND human_eval.question_id = eval_result.question_id)
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        result = [one[1:10] for one in result]
        return result

    def select_from_human_eval(self, model_name, question_id):
        query = """
            SELECT * FROM human_eval WHERE model_name = ? AND question_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (model_name, question_id))
        result = cursor.fetchone()
        return result

def statistic_acc(eval_result):
    eval_db = EvalDataBase('data/database/script.db')

    missing_steps_correct = 0
    redundant_steps_correct = 0
    duplicate_steps_correct = 0
    executable_correct = 0
    limitation_correct = 0
    complete_correct = 0
    step_order_correct = 0
    for one_eval in tqdm(eval_result):
        question_id = one_eval[0]
        model_name = one_eval[1]
        human_eval = eval_db.select_from_human_eval(model_name, question_id)

        # print(one_eval)
        if one_eval[2] == human_eval[2]:
            missing_steps_correct += 1
        if one_eval[3] == human_eval[3]:
            redundant_steps_correct += 1
        if one_eval[4] == human_eval[4]:
            duplicate_steps_correct += 1
        if one_eval[5] == human_eval[5]:
            executable_correct += 1
        if one_eval[6] == human_eval[6]:
            limitation_correct += 1
        if one_eval[7] == human_eval[7]:
            complete_correct += 1
        if one_eval[8] == human_eval[8]:
            step_order_correct += 1
    missing_steps_correct = round(missing_steps_correct / len(eval_result), 3)
    redundant_steps_correct = round(redundant_steps_correct / len(eval_result), 3)
    duplicate_steps_correct = round(duplicate_steps_correct / len(eval_result), 3)
    executable_correct = round(executable_correct / len(eval_result), 3)
    limitation_correct = round(limitation_correct / len(eval_result), 3)
    complete_correct = round(complete_correct / len(eval_result), 3)
    step_order_correct = round(step_order_correct / len(eval_result), 3)

    return missing_steps_correct, redundant_steps_correct, duplicate_steps_correct, executable_correct, limitation_correct, complete_correct, step_order_correct

if __name__ == '__main__':
    eval_db = EvalDataBase('data/database/script.db')
    model_eval = eval_db.get_model_eval()
    model_eval = statistic_acc(model_eval)
    MASEval = eval_db.get_MASEval()
    MASEval = statistic_acc(MASEval)

    print('Single Agent', model_eval)
    print('ABSEval', MASEval)
    labels = ['No Missing Steps', 'No Redundant Steps', 'No Duplicate Steps', 'Executable', 'Satisfy Constraints', 'Complete Goal', 'Correct Order']
    plot_tuples(model_eval, MASEval, labels)
