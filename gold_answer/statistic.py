import sqlite3
from proplot import rc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def select_eval_result_with_gold_answer(self):
        query = """
            SELECT * FROM whether_gold_answer WHERE with_gold_answer LIKE 'True'
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def select_eval_result_without_gold_answer(self):
        query = """
            SELECT * FROM whether_gold_answer WHERE with_gold_answer LIKE 'False'
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def select_from_human_eval(self, question_id, model_name):
        query = """
            SELECT * FROM human_eval WHERE question_id = ? AND model_name = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id, model_name))
        result = cursor.fetchone()
        return result

def plot_multiple_radar_charts(categories, values_list, titles):
    plt.rcParams.update({'font.size': 60})
    rc["font.family"] = "Times New Roman"

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 添加最后一个元素，使得雷达图闭合

    fig, ax = plt.subplots(figsize=(25, 10), subplot_kw=dict(polar=True))


    # 设置角度
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 设置雷达图的网格线样式和颜色
    ax.grid(True, linestyle='-', linewidth=1, color='black')

    # 绘制雷达图
    markers = ['o', '^']  # 圆形和三角形
    colors = ['#FA7070FF', '#7AA2E3FF']  # 自定义颜色
    for idx, (values, title) in enumerate(zip(values_list, titles)):
        # 闭合雷达图
        values += [values[0]]
        ax.plot(angles, values, label=title, color=colors[idx], marker=markers[idx], markerfacecolor='white',
             markeredgewidth=3, markersize=15, linewidth=3)


    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=40)
    # ax.spines['polar'].set_visible(False)  # 不显示极坐标最外圈的圆

    # 显示刻度标签
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '', '0.8', '1.0'], fontsize=40)


    ax.legend(loc='upper right', bbox_to_anchor=(1.8, 1), fontsize=40)
    print('radar_static.jpg')
    plt.savefig('utils/graph/radar_gold_answer.svg')
    print('plot multiple radar charts success')


def get_data():
    eval_db = EvalDataBase('data/database/script.db')
    with_gold_answer = eval_db.select_eval_result_with_gold_answer()

    missing_steps_correct = 0
    redundant_steps_correct = 0
    duplicate_steps_correct = 0
    executable_correct = 0
    limitation_correct = 0
    complete_correct = 0
    step_order_correct = 0
    for one_eval in tqdm(with_gold_answer, desc='with_gold_answer:'):
        question_id = one_eval[0]
        model_name = one_eval[1]
        human_eval = eval_db.select_from_human_eval(question_id, model_name)
        human_eval = human_eval[2:9]
        one_eval = one_eval[3:10]

        if one_eval[0] == human_eval[0]:
            missing_steps_correct += 1
        if one_eval[1] == human_eval[1]:
            redundant_steps_correct += 1
        if one_eval[2] == human_eval[2]:
            duplicate_steps_correct += 1
        if one_eval[3] == human_eval[3]:
            executable_correct += 1
        if one_eval[4] == human_eval[4]:
            limitation_correct += 1
        if one_eval[5] == human_eval[5]:
            complete_correct += 1
        if one_eval[6] == human_eval[6]:
            step_order_correct += 1

    missing_steps_correct = round(missing_steps_correct / len(with_gold_answer), 3)
    redundant_steps_correct = round(redundant_steps_correct / len(with_gold_answer), 3)
    duplicate_steps_correct = round(duplicate_steps_correct / len(with_gold_answer), 3)
    executable_correct = round(executable_correct / len(with_gold_answer), 3)
    limitation_correct = round(limitation_correct / len(with_gold_answer), 3)
    complete_correct = round(complete_correct / len(with_gold_answer), 3)
    step_order_correct = round(step_order_correct / len(with_gold_answer), 3)
    data_1 = [missing_steps_correct, redundant_steps_correct, duplicate_steps_correct, executable_correct, limitation_correct, complete_correct, step_order_correct]

    without_gold_answer = eval_db.select_eval_result_without_gold_answer()
    missing_steps_correct = 0
    redundant_steps_correct = 0
    duplicate_steps_correct = 0
    executable_correct = 0
    limitation_correct = 0
    complete_correct = 0
    step_order_correct = 0
    for one_eval in tqdm(without_gold_answer, desc='with_gold_answer:'):
        question_id = one_eval[0]
        model_name = one_eval[1]
        human_eval = eval_db.select_from_human_eval(question_id, model_name)
        human_eval = human_eval[2:9]
        one_eval = one_eval[3:10]

        if one_eval[0] == human_eval[0]:
            missing_steps_correct += 1
        if one_eval[1] == human_eval[1]:
            redundant_steps_correct += 1
        if one_eval[2] == human_eval[2]:
            duplicate_steps_correct += 1
        if one_eval[3] == human_eval[3]:
            executable_correct += 1
        if one_eval[4] == human_eval[4]:
            limitation_correct += 1
        if one_eval[5] == human_eval[5]:
            complete_correct += 1
        if one_eval[6] == human_eval[6]:
            step_order_correct += 1

    missing_steps_correct = round(missing_steps_correct / len(with_gold_answer), 3)
    redundant_steps_correct = round(redundant_steps_correct / len(with_gold_answer), 3)
    duplicate_steps_correct = round(duplicate_steps_correct / len(with_gold_answer), 3)
    executable_correct = round(executable_correct / len(with_gold_answer), 3)
    limitation_correct = round(limitation_correct / len(with_gold_answer), 3)
    complete_correct = round(complete_correct / len(with_gold_answer), 3)
    step_order_correct = round(step_order_correct / len(with_gold_answer), 3)
    data_2 = [missing_steps_correct, redundant_steps_correct, duplicate_steps_correct, executable_correct,
              limitation_correct, complete_correct, step_order_correct]


    category = ['No Missing\nSteps', 'No Redundant\nSteps', 'No Duplicate\nSteps', 'Executable', 'Satisfy\nConstraint', 'Complete\nScript', 'Step\nCorrect']
    data = [data_1, data_2]
    titles = ['With gold answer', 'Without gold answer']
    print('With gold answer:', data_1)
    print('Without gold answer:', data_2)
    plot_multiple_radar_charts(category, data, titles)





if __name__ == '__main__':
    get_data()
