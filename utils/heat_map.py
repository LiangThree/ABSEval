import sqlite3
from pprint import pprint
from proplot import rc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def select_eval_result(self):
        query = """
            SELECT DISTINCT category
            FROM question
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        category_list = [one[0] for one in result]

        model_list = [  "Baichuan-13B-Chat",
                        "Baichuan2-13B-Chat",
                        "Qwen-7B-Chat",
                        "Qwen-14B-Chat",
                        "Qwen-72B-Chat",
                        "Mistral-7B-Instruct-v0.1",
                        "Mistral-7B-Instruct-v0.2",
                        "Mistral-8x7B-Instruct-v0.1",
                        "LLaMa2-7b-chat",
                        "LLaMa2-13b-chat",
                        "LLaMa-2-70b-chat",
                        "Vicuna-7b-v1.5",
                        "Vicuna-13b-v1.5",
                        "LLaMa3-8b-instruct",
                        "LLaMa3-70b-instruct"]

        question_list = {}
        for one_category in category_list:
            question_list[one_category] = {}
            for model in model_list:
                query = f"""
                     SELECT eval_result.*
                     FROM eval_result JOIN question ON eval_result.question_id = question.question_id
                     WHERE question.category = ? and eval_result.model_name like '%{model}%'
                """
                cursor = self.conn.cursor()
                cursor.execute(query, (one_category,))
                result = cursor.fetchall()
                sum = 0

                for one_eval_result in result:
                    one_eval_result = one_eval_result[3:10]
                    if one_eval_result[0] == 'False':
                        sum += 1
                    if one_eval_result[1] == 'False':
                        sum += 1
                    if one_eval_result[2] == 'False':
                        sum += 1
                    if one_eval_result[3] == 'True':
                        sum += 1
                    if one_eval_result[4] == 'True':
                        sum += 1
                    if one_eval_result[5] == 'True':
                        sum += 1
                    if one_eval_result[6] == 'True':
                        sum += 1

                sum = sum/(7*len(result))
                sum = round(sum, 2)

                question_list[one_category][model] = sum

        plot_heat_model(question_list, 'utils/graph/model_performance_heat.svg')

    def select_size_eval_result(self):
        model_70b = ["meta/Llama-2-70b-chat", "meta/llama3-70b-instruct", "qwen/Qwen-72B-Chat"]
        model_7b = ["meta/llama2-7b-chat","meta/llama3-8b-instruct","qwen/Qwen-7B-Chat","mistralai/Mistral-7B-Instruct-v0.1","mistralai/Mistral-7B-Instruct-v0.2","lmsys/vicuna-7b-v1.5"]
        model_13b = ["meta/llama2-13b-chat","baichuan-inc/Baichuan-13B-Chat", "baichuan-inc/Baichuan2-13B-Chat", "qwen/Qwen-14B-Chat","lmsys/vicuna-13b-v1.5"]
        model_40b = ["mistralai/Mistral-8x7B-Instruct-v0.1"]

        size_eval = {
            "70b": {},
            "7b": {},
            "13b": {},
            "40b": {},
            }

        for index, model_list in enumerate([model_70b, model_7b, model_13b, model_40b]):
            model_statistic = {}
            model_statistic['Missing Steps'] = 0
            model_statistic['Redundant Steps'] = 0
            model_statistic['Duplicate Steps'] = 0
            model_statistic['Executable'] = 0
            model_statistic['Satisfy Limitation'] = 0
            model_statistic['Complete Goal'] = 0
            model_statistic['Order Correct'] = 0
            model_statistic['count'] = 0
            for model in model_list:
                query = f"""
                    SELECT eval_result.*
                    FROM eval_result JOIN question ON eval_result.question_id = question.question_id
                    WHERE eval_result.model_name like '{model}'
                """
                # print(query)
                cursor = self.conn.cursor()
                cursor.execute(query, ())
                result = cursor.fetchall()

                for one_eval_result in result:
                    one_eval_result = one_eval_result[3:10]
                    if one_eval_result[0] == 'False':
                        model_statistic['Missing Steps'] += 1
                    if one_eval_result[1] == 'False':
                        model_statistic['Redundant Steps'] += 1
                    if one_eval_result[2] == 'False':
                        model_statistic['Duplicate Steps'] += 1
                    if one_eval_result[3] == 'True':
                        model_statistic['Executable'] += 1
                    if one_eval_result[4] == 'True':
                        model_statistic['Satisfy Limitation'] += 1
                    if one_eval_result[5] == 'True':
                        model_statistic['Complete Goal'] += 1
                    if one_eval_result[6] == 'True':
                        model_statistic['Order Correct'] += 1
                    model_statistic['count'] += 1

            model_statistic['Missing Steps'] = round(model_statistic['Missing Steps'] / model_statistic['count'], 2)
            model_statistic['Redundant Steps'] = round(model_statistic['Redundant Steps'] / model_statistic['count'], 2)
            model_statistic['Duplicate Steps'] = round(model_statistic['Duplicate Steps'] / model_statistic['count'], 2)
            model_statistic['Executable'] = round(model_statistic['Executable'] / model_statistic['count'], 2)
            model_statistic['Satisfy Limitation'] = round(model_statistic['Satisfy Limitation'] / model_statistic['count'], 2)
            model_statistic['Complete Goal'] = round(model_statistic['Complete Goal'] / model_statistic['count'], 2)
            model_statistic['Order Correct'] = round(model_statistic['Order Correct'] / model_statistic['count'], 2)
            model_statistic.pop('count')

            size_eval[list(size_eval.keys())[index]] = model_statistic

        plot_heat(size_eval, 'utils/graph/heat.svg')


def plot_heat_model(data, file_name):
    # df = pd.DataFrame(data)
    # print(df)
    # 将数据字典转换为DataFrame
    df = pd.DataFrame(data).T
    rc["font.family"] = "Times New Roman"
    font_size = 50

    print(df)

    # 计算每个类别和模型的均值
    category_means = df.mean(axis=1)
    model_means = df.mean(axis=0)

    # 根据均值对行和列进行排序
    df_sorted = df.loc[category_means.sort_values(ascending=False).index, model_means.sort_values().index]

    # 设置图表大小
    plt.figure(figsize=(40, 30))

    ax = sns.heatmap(df_sorted, annot=True, cmap='YlGnBu', cbar=True, fmt='.2f',
                annot_kws={"size": font_size},  # 设置注释字体大小
    )

    # 添加标题和标签
    # 设置标签字体大小
    plt.xticks(rotation=45, fontsize=font_size)
    plt.yticks(rotation=45, fontsize=font_size)

    # 添加标题和标签
    # plt.title('Heatmap of Categories and Models (Sorted by Mean Values)', fontsize=16)
    plt.xlabel('Model', fontsize=60, labelpad=40)
    plt.ylabel('Category', fontsize=60)

    # 设置图例字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)  # 设置colorbar的字体大小

    # 显示图表
    plt.savefig(file_name)


def plot_heat(data, file_name):
    # df = pd.DataFrame(data)
    # print(df)
    # 将数据字典转换为DataFrame
    df = pd.DataFrame(data)
    rc["font.family"] = "Times New Roman"
    font_size = 50

    print(df)

    # 计算每个类别和模型的均值
    category_means = df.mean(axis=1)
    model_means = df.mean(axis=0)

    # 根据均值对行和列进行排序
    df_sorted = df.loc[category_means.sort_values(ascending=False).index, model_means.sort_values().index]

    # 设置图表大小
    plt.figure(figsize=(12, 12))

    ax = sns.heatmap(df_sorted, annot=True, cmap='YlGnBu', cbar=True, fmt='.2f',
                annot_kws={"size": font_size},  # 设置注释字体大小
                )

    # 添加标题和标签
    # 设置标签字体大小
    plt.xticks(rotation=0, fontsize=font_size)
    plt.yticks(rotation=0, fontsize=font_size)

    # 添加标题和标签
    # plt.title('Heatmap of Categories and Models (Sorted by Mean Values)', fontsize=16)
    plt.xlabel('Model Size', fontsize=font_size)
    plt.ylabel('Criteria', fontsize=font_size)

    # 设置图例字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)  # 设置 colorbar 的字体大小

    # 显示图表
    plt.savefig(file_name)

if __name__ == '__main__':
    db_path = 'data/database/script.db'
    eval_db = EvalDataBase(db_path)
    eval_db.select_size_eval_result()
    eval_db.select_eval_result()
