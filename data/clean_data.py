import re
from database.util.database_util import EvalDataBase
from tqdm import *

"""
输入一个str，删除序号1之前的内容，如果序号是7，8，9开头，将其替换为1，2，3
"""
eval_db = EvalDataBase('data/database/script.db')

def simplify_answer(input_str):

    if input_str == None:
        return

    origin_str = input_str

    for i, char in enumerate(input_str):
        if char.isdigit() and input_str[i + 1:i + 2] == '.':
            input_str = input_str[i:]
            break

    if len(input_str) == 0:
        return origin_str

    pattern = r'\b\d+\.'

    step_number = 1  # 添加了一个步骤计数器

    def replace(match):
        nonlocal step_number
        replacement = str(step_number) + '.'
        step_number += 1
        return replacement

    replaced_text = re.sub(pattern, replace, input_str)

    if len(input_str) == 0:
        return origin_str

    return replaced_text

def filter_gold_answer_without_learn():
    print('clean answer')
    answers = eval_db.select_gold_answer_without_learn()
    for item in tqdm(answers):
        question_id = item[0]
        answers = item[1]
        answers = simplify_answer(answers)
        eval_db.update_gold_answer_without_learn(answers, question_id)

def filter_gold_answer():
    print('clean gold answer')
    answers = eval_db.select_answer_from_gold_answer()
    for item in tqdm(answers):
        question_id = item[0]
        answers = item[1]
        answers = simplify_answer(answers)
        eval_db.update_answer_from_gold_answer(answers, question_id)
    clean_gold_answer()

def filter_interfere_data():
    print('clean interfere data')
    answers = eval_db.select_answer_from_interfere()
    for item in tqdm(answers):
        question_id, model_name, interfere_category, interfere = item
        interfere = simplify_answer(interfere)
        eval_db.update_interfere(interfere, question_id, model_name, interfere_category)
    clean_gold_answer()

def filter_llm_inference(eval_db:EvalDataBase):
    answers = eval_db.select_inference_from_llm_inference()
    for item in tqdm(answers):
        question_id = item[0]
        model_name = item[1]
        inference = item[2]
        inference = simplify_answer(inference)
        eval_db.update_inference_from_llm_inference(question_id, model_name, inference)

def clean_01ai_inference():
    inferences = eval_db.select_inference_of_01ai()
    for one_inference in tqdm(inferences):
        question_id, model_name, inference = one_inference
        paragraphs = inference.split('\n')
        print(inference)
        # 遍历段落列表，找到第一个有标号的段落
        inference_after_clean = ""
        for paragraph in paragraphs:
            # 检查段落是否以数字和句点开头
            if paragraph.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '11.', '12.', '13.', '14.', '15.', '16.', '17.', '18.', '19.', '20.')):
                inference_after_clean += paragraph
            else:
                print("after clean")
                print(inference_after_clean)
                print('--------------------------------------------')
                break

def clean_refuse_answer():
    answers = eval_db.select_inference_from_llm_inference()
    count = 0
    for item in tqdm(answers):
        question_id = item[0]
        model_name = item[1]
        inference = item[2]
        if inference is None:
            continue
        if inference[0] is None:
            continue
        if inference[0] != '1':
            count += 1
            eval_db.update_inference_from_llm_inference(question_id, model_name, None)
    print(f'Number of questions refused to be answered:{count}')


def clean_gold_answer():
    answers = eval_db.select_gold_answer()
    count = 0
    for item in tqdm(answers):
        model_name = item[0]
        question_id = item[1]
        gold_answer = item[2]
        if gold_answer is None:
            continue
        if gold_answer[0] is None:
            continue
        if gold_answer[0] != '1':
            count += 1
            eval_db.delete_from_gold_answer(model_name, question_id)
    print(f'Number of questions refused to be answered:{count}')

def clean_question_less_limitation():
    eval_db.clean_question_less_limitation()


if __name__ == '__main__':

    # clean_question_less_limitation()
    # filter_gold_answer()
    filter_gold_answer_without_learn()
    # filter_llm_inference()
    # clean_01ai_inference()
    # clean_refuse_answer()
    # filter_interfere_data()
