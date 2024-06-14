import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *

class EvalDataBase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def select_frequency_from_abstract_question(self, category_list):
        all_frequency = []
        for category in category_list:
            cursor = self.conn.cursor()
            query = f'''
            SELECT question_id, frequency
            FROM abstract_question
            WHERE category like ?
            '''
            cursor.execute(query, (category,))
            current_frequency = cursor.fetchall()
            for item in current_frequency:
                all_frequency.append(item)
        return all_frequency

    def select_frequency_from_question(self, category_list):
        all_frequency = []
        for category in category_list:
            cursor = self.conn.cursor()
            query = f'''
            SELECT question_id, frequency
            FROM question
            WHERE category like ?
            '''
            cursor.execute(query, (category,))
            current_frequency = cursor.fetchall()
            for item in current_frequency:
                all_frequency.append(item)
        return all_frequency

    def update_question_frequency(self, question_id, frequency_choice):
        cursor = self.conn.cursor()
        query = f'''
            UPDATE question
            SET frequency = ?
            WHERE question_id = ?
        '''
        cursor.execute(query, (frequency_choice, question_id))
        self.conn.commit()

    def close(self):
        self.conn.close()

def get_frequency_statistic(frequencies):
    frequencies = [int(freq.replace(",", "")) for freq in frequencies]

    frequencies.sort()

    # 设置合适的范围和跨度
    min_freq = min(frequencies)
    max_freq = max(frequencies)
    bin_width = 200000  # 设置跨度为 50000

    # 计算直方图的边界
    bins = range(min_freq, max_freq + bin_width, bin_width)

    # 绘制直方图
    plt.hist(frequencies, bins=bins, edgecolor='black')
    plt.xlabel('Frequency')
    plt.ylabel('Count')
    plt.title(f'Frequency statistic')
    plt.xticks(range(min_freq, max_freq + bin_width, bin_width))
    plt.savefig('utils/graph/frequency_static.png')

def get_high_and_low_frquency(frequency_list):
    data = [int(num.replace(",", "")) for num in frequency_list]
    data.sort()
    data_length = len(data)
    first_boundary_index = data_length // 3
    second_boundary_index = first_boundary_index * 2

    print("first_boundary_index:", data[first_boundary_index])
    print("first_boundary_index:", data[second_boundary_index])

    return data[first_boundary_index], data[second_boundary_index]

if __name__ == '__main__':
    db = EvalDataBase('data/database/script.db')

    all_category = ["Arts-and-Entertainment", "Computers-and-Electronics", "Education-and-Communications",
                    "Food-and-Entertaining", "Finance-and-Business", "Health", "Hobbies-and-Crafts",
                    "Holidays-and-Traditions", "Home-and-Garden", "Personal-Care-and-Style", "Pets-and-Animals",
                    "Philosophy-and-Religion", "Relationships", "Sports-and-Fitness", "Travel"]
    category_list = all_category
    category_list = [category.replace('-', ' ') for category in category_list]

    # origin_db = EvalDataBase('database_copy/script.db')
    # frequencies = origin_db.select_frequency_from_abstract_question(category_list)
    # for question_id, frequency in tqdm(frequencies, desc='update frequency'):
    #     db.update_question_frequency(question_id, frequency)


    frequency_list = db.select_frequency_from_abstract_question(category_list)
    frequency_list = [item[1] for item in frequency_list]
    # get_frequency_statistic(frequency_list)
    first_boundary_index, second_boundary_index = get_high_and_low_frquency(frequency_list)
    question_id_and_frequency = db.select_frequency_from_question(category_list)
    for question_id, frequency in tqdm(question_id_and_frequency, desc='update frequency'):
        if frequency in ['high', "low", "medium"]:
            continue
        frequency = int(frequency.replace(",", ""))
        if frequency > second_boundary_index:
            db.update_question_frequency(question_id, 'high')
        elif frequency < first_boundary_index:
            db.update_question_frequency(question_id, 'low')
        else:
            db.update_question_frequency(question_id, 'medium')



