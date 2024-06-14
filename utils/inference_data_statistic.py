import sqlite3
import matplotlib.pyplot as plt
import numpy as np

class EvalDataBase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
    def select_question_ids_from_llm_inference(self):
        cursor = self.conn.cursor()
        query = f'''
        SELECT question_id, model_name, inference
        FROM llm_inference
        '''
        cursor.execute(query, )
        return cursor.fetchall()


def plot_histogram(data, bin_size=1000):
    # 计算最小值和最大值，以确定绘图范围
    min_val = min(data)
    max_val = max(data)


    # 计算区间数量
    num_bins = int((max_val - min_val) / bin_size) + 1

    # 计算每个区间的范围
    bins = [(min_val + i * bin_size, min_val + (i + 1) * bin_size) for i in range(num_bins)]

    # 统计每个区间中元素的数量
    bin_counts = [sum(1 for x in data if bin[0] <= x < bin[1]) for bin in bins]

    # 绘制直方图
    plt.figure(figsize=(10, 6))  # 调整图形尺寸
    plt.bar(np.arange(len(bins)), bin_counts, align='center', color='blue', edgecolor='gray')  # 调整柱子颜色和边框颜色
    plt.xticks(np.arange(len(bins)), [f'{bin[0]}-{bin[1]}' for bin in bins], rotation='vertical', fontsize=10)  # 调整刻度样式
    plt.yticks(fontsize=10)
    plt.xlabel('length', fontsize=12)  # 调整标签样式
    plt.ylabel('number', fontsize=12)
    plt.title('inference length statics', fontsize=14)  # 调整标题样式
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加水平虚线网格线
    plt.tight_layout()  # 调整布局
    plt.savefig('utils/graph/inference_data_static.png')
    plt.show()


def statistical_inference_length(db:EvalDataBase):
    inference_data = db.select_question_ids_from_llm_inference()
    inference_length = [len(inference[2]) for inference in inference_data]
    plot_histogram(inference_length, 500)


if __name__ == '__main__':
    db = EvalDataBase("data/database/script.db")
    statistical_inference_length(db)

