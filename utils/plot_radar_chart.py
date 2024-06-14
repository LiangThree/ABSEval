import numpy as np
import matplotlib.pyplot as plt
import json
import sqlite3
import random
import os
from pprint import pprint
# from matplotlib import rc
import matplotlib.colors as mcolors
import yaml
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from proplot import rc
import numpy as np
import matplotlib.pyplot as plt

"""----------------------------------- 数据库工具 —------------------------------------------------"""
class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def get_eval_result(self):
        query = """
            SELECT * FROM eval_result
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def get_limitation_eval_result(self):
        query = """
            SELECT qa.target_view, er.*
            FROM eval_result AS er JOIN question AS qa
            WHERE er.question_id = qa.question_id
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def get_answer_length_eval_result(self):
        query = """
            SELECT DISTINCT answer_length
            FROM question
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        answer_length_result = cursor.fetchall()
        answer_length_result = [int(item[0]) for item in answer_length_result if item[0] is not None and int(item[0])>=3 ]

        query = """
            SELECT qa.answer_length, er.*
            FROM eval_result AS er JOIN question AS qa
            WHERE er.question_id = qa.question_id
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return answer_length_result, result

"""----------------------------------- 雷达图 实验1.3 —------------------------------------------------"""
def plot_multiple_radar_charts(categories, values_list, titles):
    plt.rcParams.update({'font.size': 20})
    rc["font.family"] = "Times New Roman"

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 添加最后一个元素，使得雷达图闭合

    fig, ax = plt.subplots(figsize=(30, 13), subplot_kw=dict(polar=True))

    # 设置角度
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 设置雷达图的网格线样式和颜色
    ax.grid(True, linestyle='-', linewidth=1, color='black')

    # 绘制雷达图
    for values, title in zip(values_list, titles):
        # 闭合雷达图
        values += [values[0]]
        ax.plot(angles, values, linewidth=3, label=title)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=32)
    # ax.spines['polar'].set_visible(False)  # 不显示极坐标最外圈的圆


    ax.legend(loc='upper right', bbox_to_anchor=(1.8, 1), fontsize=32)
    print('radar_static.jpg')
    plt.savefig('utils/graph/radar_static.svg')
    print('plot multiple radar charts success')

"""----------------------------------- 所有模型在不同指标上的表现 实验1.2  —------------------------------------------------"""
def plot_histograms(categories, values_list, model_names):
    plt.rcParams.update({'font.size': 20})
    rc["font.family"] = "Times New Roman"
    rc["axes.labelsize"] = 20

    num_models = len(model_names)

    for i, (category, values, title) in enumerate(zip(categories, values_list, model_names)):
        fig, ax = plt.subplots(figsize=(10, 6))

        sorted_indices = np.argsort(values)[::-1]  # Sort indices by descending values
        sorted_values = [values[idx] for idx in sorted_indices]
        sorted_titles = [list(model_names)[idx] for idx in sorted_indices]

        colors = []
        for label in sorted_titles:
            if 'llama3' in label.lower():
                color_list = ['#EF4F4FFF', '#EF4F4FBF']
                if label == 'llama3-70b-instruct':
                    colors.append(color_list[0])
                if label == 'llama3-8b-instruct':
                    colors.append(color_list[1])
            if 'llama' in label.lower():
                color_list = ['#FA7070FF', '#FA7070BF', '#FA707080', '#FA707040', '#FA707020']
                if label == 'Llama-2-70b-chat':
                    colors.append(color_list[0])
                if label == 'llama2-13b-chat':
                    colors.append(color_list[1])
                if label == 'llama2-7b-chat':
                    colors.append(color_list[2])
            elif 'mistral' in label.lower():
                color_list = ['#7AA2E3FF', '#7AA2E3BF', '#7AA2E380']
                if label == 'Mistral-8x7B-Instruct-v0.1':
                    colors.append(color_list[0])
                if label == 'Mistral-7B-Instruct-v0.2':
                    colors.append(color_list[1])
                if label == 'Mistral-7B-Instruct-v0.1':
                    colors.append(color_list[2])
            elif 'qwen' in label.lower():
                color_list = ['#52D3D8FF', '#52D3D8BF', '#52D3D880']
                if label == 'Qwen-72B-Chat':
                    colors.append(color_list[0])
                if label == 'Qwen-14B-Chat':
                    colors.append(color_list[1])
                if label == 'Qwen-7B-Chat':
                    colors.append(color_list[2])
            elif 'baichuan' in label.lower():
                color_list = ['#FFE699FF', '#FFE699BF']
                if label == 'Baichuan2-13B-Chat':
                    colors.append(color_list[0])
                if label == 'Baichuan-13B-Chat':
                    colors.append(color_list[1])
            elif 'vicuna' in label.lower():
                color_list = ['#B2A4FFFF', '#B2A4FFBF']
                if label == 'vicuna-13b-v1.5':
                    colors.append(color_list[0])
                if label == 'vicuna-7b-v1.5':
                    colors.append(color_list[1])

        x = np.arange(num_models)
        bars = ax.bar(x, sorted_values, color=colors)

        for bar, value in zip(bars, sorted_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{value:.2f}', ha='center', va='bottom')

        if category != 'c. No duplicate steps':
            # Adjust the y-axis to not start from zero
            ymin = min(sorted_values) * 0.95
            ymax = max(sorted_values) * 1.05
            ax.set_ylim(ymin, ymax)

        ax.set_xticks(x)
        sorted_titles = replace_model_name(sorted_titles)
        ax.set_xticklabels(sorted_titles, rotation=45, ha='right')
        ax.set_ylabel('ACC')
        ax.set_title(category)

        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

        plt.tight_layout()
        name = category.replace(' ', '_')
        plt.savefig(f'utils/graph/category/histogram_{name}.svg')

"""----------------------------------- 所有指标的表现 实验1.1 —------------------------------------------------"""
def plot_bar_chart(data_dict, title):
    # plt.style.use('seaborn-darkgrid')  # 使用seaborn的darkgrid样式，深色背景更显眼

    # 统一设置字体
    rc["font.family"] = "Times New Roman"
    # 统一设置轴刻度标签的字体大小
    rc['tick.labelsize'] = 20
    # 统一设置xy轴名称的字体大小
    rc["axes.labelsize"] = 20
    # 统一设置轴刻度标签的字体粗细
    rc["axes.labelweight"] = "light"
    # 统一设置xy轴名称的字体粗细
    rc["tick.labelweight"] = "bold"

    fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100, facecolor="w")
    fig.subplots_adjust(left=0.2, bottom=0.2)

    font2 = {'weight': 'normal',
             'size': 20, }

    labels = list(data_dict.keys())
    values = list(data_dict.values())

    # 设置网格线样式
    plt.rcParams['grid.color'] = '0.75'  # 网格线颜色
    plt.rcParams['grid.linestyle'] = '--'  # 网格线样式，这里为虚线
    plt.rcParams['grid.linewidth'] = 0.8  # 网格线宽度



    # 创建渐变颜色映射
    # base_color = '#52D3D8FF'
    # colormap = create_gradient_colormap(base_color, len(labels))
    # colors = [colormap(i / len(labels)) for i in range(len(labels))]

    colormap = plt.cm.get_cmap('viridis')
    colors = [colormap(i / len(labels)) for i in range(len(labels))]

    bars = plt.barh(labels, values, color=colors)
    plt.xlabel('Correct count', font2)
    plt.title(title, font2)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width}', ha='left', va='center', fontsize=20)

    title = title.replace(' ', '_')
    print(f'{title}.jpg')
    plt.savefig(f'utils/graph/{title}.svg')
    # plt.savefig(f'utils/graph/{title}.jpg')

"""----------------------------------- 添加限制后模型在不同指标上的表现 实验1.4 —------------------------------------------------"""
def plot_limitation_chart(data_dict, title):
    # plt.style.use('seaborn-darkgrid')  # 使用seaborn的darkgrid样式，深色背景更显眼

    # 统一设置字体
    rc["font.family"] = "Times New Roman"
    # 统一设置轴刻度标签的字体大小
    rc['tick.labelsize'] = 20
    # 统一设置xy轴名称的字体大小
    rc["axes.labelsize"] = 20
    # 统一设置轴刻度标签的字体粗细
    rc["axes.labelweight"] = "light"
    # 统一设置xy轴名称的字体粗细
    rc["tick.labelweight"] = "bold"

    fig, axes = plt.subplots(1, 1, figsize=(10, 3), dpi=100, facecolor="w")
    fig.subplots_adjust(left=0.2, bottom=0.2)

    font2 = {'weight': 'normal',
             'size': 20, }

    labels = list(data_dict.keys())
    values = list(data_dict.values())

    # 设置网格线样式
    plt.rcParams['grid.color'] = '0.75'  # 网格线颜色
    plt.rcParams['grid.linestyle'] = '--'  # 网格线样式，这里为虚线
    plt.rcParams['grid.linewidth'] = 0.8  # 网格线宽度


    # plt.figure(figsize=(10, 3))
    colors = ["#FA7070BF", "#7AA2E3FF", '#FFE699FF']
    bars = plt.barh(labels, values, color=colors)
    plt.xlabel('ACC', font2)
    plt.title(title, font2)


    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width}', ha='left', va='center', fontsize=20)

    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')

    title = title.replace(' ', '_')
    # print(f'{title}.jpg')
    plt.savefig(f'utils/graph/limitation/{title}.svg')


def statistic_criteria_accuray():
    eval_data_base = EvalDataBase('data/database/script.db')
    eval_result = eval_data_base.get_eval_result()
    no_missing_steps = 0
    no_redundant_steps = 0
    no_duplicate_steps = 0
    can_execute = 0
    satisfy_limitation = 0
    satisfy_complete = 0
    step_order_correct = 0
    count = 0
    for one_eval_result in eval_result:
        # print(one_eval_result)
        eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain_1, explain_2 = one_eval_result
        if missing_steps == 'False':
            no_missing_steps += 1
        if redundant_steps == 'False':
            no_redundant_steps += 1
        if duplicate_steps == 'False':
            no_duplicate_steps += 1
        if executable == 'True':
            can_execute += 1
        if limitation == 'True':
            satisfy_limitation += 1
        if complete == 'True':
            satisfy_complete += 1
        if step_order == 'True':
            step_order_correct += 1
        count += 1

    data = {
        'No Missing Steps: ': round(no_missing_steps/count, 3),
        'No Redundant Steps: ': round(no_redundant_steps/count, 3),
        'No Duplicate Steps: ': round(no_duplicate_steps/count, 3),
        'Executable: ': round(can_execute/count, 3),
        'Satisfy Constraint: ': round(satisfy_limitation/count, 3),
        'Complete Goal: ': round(satisfy_complete/count, 3),
        'Order Correct: ': round(step_order_correct/count, 3)
    }

    print(data)
    plot_bar_chart(data, 'Data Statistics')


def get_radar():
    eval_data_base = EvalDataBase('data/database/script.db')
    eval_result = eval_data_base.get_eval_result()
    model_name_dict = {'Qwen':{}, 'Mistral':{}, 'Baichuan':{}, 'LLaMa2':{}, 'LLaMa3':{}, 'Vicuna':{}}

    for model_type in model_name_dict.keys():
        model_name_dict[model_type] = {}
        model_name_dict[model_type]['missing_steps'] = 0
        model_name_dict[model_type]['redundant_steps'] = 0
        model_name_dict[model_type]['duplicate_steps'] = 0
        model_name_dict[model_type]['can_execute'] = 0
        model_name_dict[model_type]['satisfy_limitation'] = 0
        model_name_dict[model_type]['satisfy_complete'] = 0
        model_name_dict[model_type]['step_order_correct'] = 0
        model_name_dict[model_type]['count'] = 0


    for one_eval_result in tqdm(eval_result):
        eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain_1, explain_2 = one_eval_result
        for model_type in model_name_dict:
            if model_type.lower() in model_name.lower():
                if missing_steps == 'False':
                    model_name_dict[model_type]['missing_steps'] += 1
                if redundant_steps == 'False':
                    model_name_dict[model_type]['redundant_steps'] += 1
                if duplicate_steps == 'False':
                    model_name_dict[model_type]['duplicate_steps'] += 1
                if executable == 'True':
                    model_name_dict[model_type]['can_execute'] += 1
                if limitation == 'True':
                    model_name_dict[model_type]['satisfy_limitation'] += 1
                if complete == 'True':
                    model_name_dict[model_type]['satisfy_complete'] += 1
                if step_order == 'True':
                    model_name_dict[model_type]['step_order_correct'] += 1
                model_name_dict[model_type]['count'] += 1

    for model_type in model_name_dict.keys():
        model_name_dict[model_type]['missing_steps'] = round(model_name_dict[model_type]['missing_steps']/model_name_dict[model_type]['count'],3)
        model_name_dict[model_type]['redundant_steps'] = round(model_name_dict[model_type]['redundant_steps']/model_name_dict[model_type]['count'],3)
        model_name_dict[model_type]['duplicate_steps'] = round(model_name_dict[model_type]['duplicate_steps']/model_name_dict[model_type]['count'],3)
        model_name_dict[model_type]['can_execute'] = round(model_name_dict[model_type]['can_execute']/model_name_dict[model_type]['count'],3)
        model_name_dict[model_type]['satisfy_limitation'] = round(model_name_dict[model_type]['satisfy_limitation']/model_name_dict[model_type]['count'],3)
        model_name_dict[model_type]['satisfy_complete'] = round(model_name_dict[model_type]['satisfy_complete']/model_name_dict[model_type]['count'],3)
        model_name_dict[model_type]['step_order_correct'] = round(model_name_dict[model_type]['step_order_correct']/model_name_dict[model_type]['count'],3)
        model_name_dict[model_type].pop('count')

    titles = model_name_dict.keys()
    values_list = []
    for model_name in titles:
        current_value = model_name_dict[model_name]
        values_list.append(list(current_value.values()))

    categories = ['No missing\nsteps', 'No redundant\nsteps', 'No duplicate\nsteps', 'Executable',
                  'Satisfy\nlimitation', 'Complete\nscript', 'Step\ncorrect']
    print(categories, values_list, titles)
    plot_multiple_radar_charts(categories, values_list, titles)


def get_data():
    eval_data_base = EvalDataBase('data/database/script.db')
    eval_result = eval_data_base.get_eval_result()

    model_name_dict = {}
    print("getting data ...")

    for one_eval_result in tqdm(eval_result):
        eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain_1, explain_2 = one_eval_result
        if model_name in model_name_dict:
            if missing_steps == 'False':
                model_name_dict[model_name]['missing_steps'] += 1
            if redundant_steps == 'False':
                model_name_dict[model_name]['redundant_steps'] += 1
            if duplicate_steps == 'False':
                model_name_dict[model_name]['duplicate_steps'] += 1
            if executable == 'True':
                model_name_dict[model_name]['can_execute'] += 1
            if limitation == 'True':
                model_name_dict[model_name]['satisfy_limitation'] += 1
            if complete == 'True':
                model_name_dict[model_name]['satisfy_complete'] += 1
            if step_order == 'True':
                model_name_dict[model_name]['step_order_correct'] += 1

            model_name_dict[model_name]['count'] += 1
        else:
            model_name_dict[model_name] = {}
            model_name_dict[model_name]['missing_steps'] = 0
            model_name_dict[model_name]['redundant_steps'] = 0
            model_name_dict[model_name]['duplicate_steps'] = 0
            model_name_dict[model_name]['can_execute'] = 0
            model_name_dict[model_name]['satisfy_limitation'] = 0
            model_name_dict[model_name]['satisfy_complete'] = 0
            model_name_dict[model_name]['step_order_correct'] = 0
            model_name_dict[model_name]['count'] = 1

            if missing_steps == 'False':
                model_name_dict[model_name]['missing_steps'] += 1
            if redundant_steps == 'False':
                model_name_dict[model_name]['redundant_steps'] += 1
            if duplicate_steps == 'False':
                model_name_dict[model_name]['duplicate_steps'] += 1
            if executable == 'True':
                model_name_dict[model_name]['can_execute'] += 1
            if limitation == 'True':
                model_name_dict[model_name]['satisfy_limitation'] += 1
            if complete == 'True':
                model_name_dict[model_name]['satisfy_complete'] += 1
            if step_order == 'True':
                model_name_dict[model_name]['step_order_correct'] += 1

    for model_name in model_name_dict.keys():
        model_name_dict[model_name]['missing_steps'] = round(model_name_dict[model_name]['missing_steps']/model_name_dict[model_name]['count'],3)
        model_name_dict[model_name]['redundant_steps'] = round(model_name_dict[model_name]['redundant_steps']/model_name_dict[model_name]['count'],3)
        model_name_dict[model_name]['duplicate_steps'] = round(model_name_dict[model_name]['duplicate_steps']/model_name_dict[model_name]['count'],3)
        model_name_dict[model_name]['can_execute'] = round(model_name_dict[model_name]['can_execute']/model_name_dict[model_name]['count'],3)
        model_name_dict[model_name]['satisfy_limitation'] = round(model_name_dict[model_name]['satisfy_limitation']/model_name_dict[model_name]['count'],3)
        model_name_dict[model_name]['satisfy_complete'] = round(model_name_dict[model_name]['satisfy_complete']/model_name_dict[model_name]['count'],3)
        model_name_dict[model_name]['step_order_correct'] = round(model_name_dict[model_name]['step_order_correct']/model_name_dict[model_name]['count'],3)
        model_name_dict[model_name].pop('count')
    # pprint(model_name_dict)

    titles = model_name_dict.keys()
    model_names = []
    for i, model_name in enumerate(titles):
        index = model_name.find('/')
        if index != -1:
            model_names.append(model_name[index + 1:])
        else:
            model_names.append(model_names)
    # print(model_names)
    values_list = []

    for model_name in titles:
        current_value = model_name_dict[model_name]
        values_list.append(list(current_value.values()))


    categories = ['a. No Missing Steps', 'b. No Redundant Steps', 'c. No Duplicate Steps', 'd. Executable', 'e. Satisfy Constraints',
                  'f. Complete Script', 'g. Step Correct']
    plot_histograms(categories, np.transpose(values_list), model_names)







def replace_model_name(model_names):
    for index, model_name in enumerate(model_names):
        if model_name == 'vicuna-13b-v1.5':
            model_names[index] = 'Vicuna-13b-v1.5'
        if model_name == 'llama2-13b-chat':
            model_names[index] = 'LLaMa2-13b-Chat'
        if model_name == 'llama3-70b-instruct':
            model_names[index] = 'LLaMa3-70b-Instruct'
        if model_name == 'llama3-8b-instruct':
            model_names[index] = 'LLaMa3-8b-Instruct'
        if model_name == 'vicuna-13b-v1.5':
            model_names[index] = 'vicuna-7b-v1.5'
        if model_name == 'llama2-7b-chat':
            model_names[index] = 'LLaMa2-7b-chat'
        if model_name == 'Llama-2-70b-chat':
            model_names[index] = 'LLaMa2-70b-Chat'
    return model_names

def get_limitation_data():
    eval_data_base = EvalDataBase('data/database/script.db')
    eval_result = eval_data_base.get_limitation_eval_result()

    limitation_data = []

    for limitation_num in range(1, 4):

        no_missing_steps = 0
        no_redundant_steps = 0
        no_duplicate_steps = 0
        can_execute = 0
        satisfy_limitation = 0
        satisfy_complete = 0
        step_order_correct = 0

        for one_eval_result in eval_result:
            target_view, eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain_1, explain_2 = one_eval_result
            # print(target_view, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain_1, explain_2)

            if int(target_view) == int(limitation_num):
                if missing_steps == 'False':
                    no_missing_steps += 1
                if redundant_steps == 'False':
                    no_redundant_steps += 1
                if duplicate_steps == 'False':
                    no_duplicate_steps += 1
                if executable == 'True':
                    can_execute += 1
                if limitation == 'True':
                    satisfy_limitation += 1
                if complete == 'True':
                    satisfy_complete += 1
                if step_order == 'True':
                    step_order_correct += 1
        data = {
            'no missing steps: ': no_missing_steps,
            'no redundant steps: ': no_redundant_steps,
            'no duplicate_steps: ': no_duplicate_steps,
            'executable: ': can_execute,
            'satisfy limitation: ': satisfy_limitation,
            'complete script: ': satisfy_complete,
            'step correct: ': step_order_correct
        }
        # print(f"limitation:{limitation_num}")
        # print(data)
        limitation_data.append(list(data.values()))

    # print(limitation_data)
    # limitation_data = np.transpose(limitation_data)

    for index, category in enumerate(['No missing steps', 'No redundant steps', 'No duplicate steps', 'Executable', 'Satisfy limitation', 'Complete script', 'Step correct']):
        limitation_dict = {}
        limitation_dict['limitation 1'] = limitation_data[0][index]
        limitation_dict['limitation 2'] = limitation_data[1][index]
        limitation_dict['limitation 3'] = limitation_data[2][index]
        # print(limitation_dict)
        plot_limitation_chart(limitation_dict, f'{category} limitation data statistics')


def adjust_color_brightness(color, factor):
    """Adjust the brightness of the given color by a factor between 0 and 1."""
    color = mcolors.to_rgba(color)
    adjusted_color = [min(1, c * factor) for c in color[:3]] + [color[3]]
    return adjusted_color


def create_gradient_colormap(base_color, n_colors=256):
    """Create a colormap that varies from dark to light based on the base color."""
    base_color = mcolors.to_rgb(base_color)
    dark_color = [c * 0.5 for c in base_color]  # Darker version of the base color
    light_color = [1 - (1 - c) * 0.5 for c in base_color]  # Lighter version of the base color

    colors = np.linspace(dark_color, light_color, n_colors // 2).tolist() + \
             np.linspace(light_color, dark_color, n_colors // 2).tolist()
    return mcolors.LinearSegmentedColormap.from_list("custom_gradient", colors)

def plot_answer_length(data, title):
    plt.rcParams.update({'font.size': 50})

    # 统一设置字体
    rc["font.family"] = "Times New Roman"
    # 统一设置轴刻度标签的字体大小
    rc['tick.labelsize'] = 50
    # 统一设置xy轴名称的字体大小
    rc["axes.labelsize"] = 50
    # 统一设置轴刻度标签的字体粗细
    rc["axes.labelweight"] = "light"
    # 统一设置xy轴名称的字体粗细
    rc["tick.labelweight"] = "bold"

    # 设置每个小字典的键作为直方图的标签
    labels = list(data.keys())


    # 提取正确和错误数量
    correct_values = [value['correct'] for value in data.values()]
    error_values = [value['error'] for value in data.values()]
    acc_values = []
    for correct, error in zip(correct_values, error_values):
        acc_values.append(round(correct/(correct+error), 3))

    plt.figure(figsize=(25, 16))

    indices_to_remove = [i for i, num in enumerate(labels) if num < 5 or num > 21]
    labels = [num for i, num in enumerate(labels) if i not in indices_to_remove]
    acc_values = [num for i, num in enumerate(acc_values) if i not in indices_to_remove]

    # 创建渐变颜色映射
    colormap = plt.cm.get_cmap('viridis')
    colors = [colormap(i / len(labels)) for i in range(len(labels))]

    # 绘制直方图
    bars = plt.bar(labels, acc_values, color=colors)
    plt.xticks(labels)


    for acc, bar in zip(acc_values, bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{acc:.2f}', ha='center', va='bottom', fontsize=40)

    # 添加图例
    # plt.legend()

    # 添加标签和标题
    plt.xlabel('Step length')
    plt.ylabel('LLM performance')
    plt.title(f'{title}')
    print(f'plot answer_length_{title}.jpg')

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # print(data)
    plt.savefig(f'utils/graph/answer_length/answer_length_{title}.svg')

def get_answer_length_data():
    eval_data_base = EvalDataBase('data/database/script.db')
    answer_length_result, eval_result = eval_data_base.get_answer_length_eval_result()
    print(answer_length_result)

    answer_length_result_dict = {}
    answer_length_transpose_dict = {}
    answer_length_transpose_dict['No Missing Steps'] = {}
    answer_length_transpose_dict['No Redundant Steps'] = {}
    answer_length_transpose_dict['No Duplicate Steps'] = {}
    answer_length_transpose_dict['Executable'] = {}
    answer_length_transpose_dict['Satisfy Constraints'] = {}
    answer_length_transpose_dict['Complete Goal'] = {}
    answer_length_transpose_dict['Correct Order'] = {}

    for one_answer_length in answer_length_result:
        answer_length_result_dict['No Missing Steps'] = {'correct': 0, 'error':0}
        answer_length_result_dict['No Redundant Steps'] = {'correct': 0, 'error':0}
        answer_length_result_dict['No Duplicate Steps'] = {'correct': 0, 'error':0}
        answer_length_result_dict['Executable'] = {'correct': 0, 'error':0}
        answer_length_result_dict['Satisfy Constraints'] = {'correct': 0, 'error':0}
        answer_length_result_dict['Complete Goal'] = {'correct': 0, 'error':0}
        answer_length_result_dict['Correct Order'] = {'correct': 0, 'error':0}


        for one_eval_result in eval_result:
            answer_length, eval_model_name, question_id, model_name, missing_steps, redundant_steps, duplicate_steps, executable, limitation, complete, step_order, explain_1, explain_2 = one_eval_result

            if int(one_answer_length) == int(answer_length):
                if missing_steps == 'False':
                    answer_length_result_dict['No Missing Steps']['correct'] += 1
                elif missing_steps == 'True':
                    answer_length_result_dict['No Missing Steps']['error'] += 1

                if redundant_steps == 'False':
                    answer_length_result_dict['No Redundant Steps']['correct'] += 1
                elif redundant_steps == 'True':
                    answer_length_result_dict['No Redundant Steps']['error'] += 1

                if duplicate_steps == 'False':
                    answer_length_result_dict['No Duplicate Steps']['correct'] += 1
                elif duplicate_steps == 'True':
                    answer_length_result_dict['No Duplicate Steps']['error'] += 1

                if executable == 'True':
                    answer_length_result_dict['Executable']['correct'] += 1
                elif executable == 'False':
                    answer_length_result_dict['Executable']['error'] += 1

                if limitation == 'True':
                    answer_length_result_dict['Satisfy Constraints']['correct'] += 1
                elif limitation == 'False':
                    answer_length_result_dict['Satisfy Constraints']['error'] += 1

                if complete == 'True':
                    answer_length_result_dict['Complete Goal']['correct'] += 1
                elif complete == 'False':
                    answer_length_result_dict['Complete Goal']['error'] += 1

                if step_order == 'True':
                    answer_length_result_dict['Correct Order']['correct'] += 1
                elif step_order == 'False':
                    answer_length_result_dict['Correct Order']['error'] += 1


        # plot_answer_length(answer_length_result_dict, str(one_answer_length))
        for key in answer_length_result_dict.keys():
            answer_length_transpose_dict[key][one_answer_length] = answer_length_result_dict[key]


    # remove_list = []
    #
    # for key in answer_length_transpose_dict.keys():
    #     for one_answer_length in answer_length_transpose_dict[key].keys():
    #         if answer_length_transpose_dict[key][one_answer_length]["correct"] + answer_length_transpose_dict[key][one_answer_length]["correct"] < 70:
    #             remove_list.append(one_answer_length)
    #
    #     for remove_key in remove_list:
    #         answer_length_transpose_dict[key].pop(str(remove_key))


    # for key in answer_length_transpose_dict.keys():
    #     print(key)
    #     print('|criteria|correct|error|sum|acc|')
    #     print('|--------|-------|-----|---|---|')
    #     for one_answer_length in answer_length_transpose_dict[key].keys():
    #         sum = answer_length_transpose_dict[key][one_answer_length]["correct"] + answer_length_transpose_dict[key][one_answer_length]["error"]
    #         acc = round(answer_length_transpose_dict[key][one_answer_length]["correct"] / sum, 3)
    #         print(f'|{one_answer_length}|{answer_length_transpose_dict[key][one_answer_length]["correct"]}|{answer_length_transpose_dict[key][one_answer_length]["error"]}|{sum}|{acc}|')


    for key in answer_length_transpose_dict.keys():
        data = answer_length_transpose_dict[key]
        plot_answer_length(data, str(key))



if __name__ == '__main__':
    # get_radar()
    # statistic_criteria_accuray()
    # get_data()
    # get_limitation_data()
    get_answer_length_data()


