import json
import random
from database_util import EvalDataBase
from tqdm import *

"""
从coscript中随机抽取数组作为我们数据集的一部分
"""

db = EvalDataBase("./data/database/script.db")

# 读取 JSON 文件
with open('./data/database/coscript/Dataset/train_data.json', 'r', encoding='utf-8') as file:
    data = file.readlines()

# 将 JSON 数据转换为 Python 字典列表
json_data = [json.loads(line.strip()) for line in data]

# 从数据中随机选择 100 条
random_selection = random.sample(json_data, 30)

# 打印选中的条目
for item in tqdm(random_selection):
    question_id = '1'+str('{:06d}'.format(item['id']))
    question = item['Specific Goal']
    category = item['Category']
    difficult_level = '1'
    limitation = str([item['Constraint']])
    answer = str(item['Script'])
    db.insert_into_question((question_id, category, difficult_level, limitation, question, answer))


