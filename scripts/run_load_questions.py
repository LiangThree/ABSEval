import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import warnings
from typing import List
from pathlib import Path
from data.database.util.database_util import EvalDataBase


def load_questions(questions_path: str):
    """load questions from jsonl file"""
    with open(questions_path, 'r') as f:
        questions = []
        for line in f:
            question = json.loads(line)
            questions.append(question)
    return questions


def load_chocies_text_and_gold_tag(question: dict):
    # load choices_text and gold_tag from question dict
    choices = question['options']
    if len(choices) == 0: return None, None
    keys = ['A', 'B', 'C', 'D']
    choices = {key: choice for key, choice in zip(keys, choices)}
    choices_text = json.dumps(choices, ensure_ascii=False)
    for key in keys:
        if question['object'] == choices[key]:
            gold_tag = key
            break
    return choices_text, gold_tag


def store_questions_to_db(questions: List[dict], target_view: str, db: EvalDataBase, question_type: str):
    """
    store questions to db, question_type appoint the table name
    question_type: choices_question or qa_question
    target_view is a column in the table
    """
    print(f'Storing {question_type} with {target_view} questions to db...')
    for i, question in enumerate(questions):
        # get meta data of question
        question_id = f'{question_type}-{target_view}-{i}'
        template_id = question['prompt-format']
        knowledge_id = '-'.join([question['subject'], question['predicate'], question['object']])
        target_view = target_view
        question_text = question['prompt']
        
        # store question to db with different table by question_type
        if question_type == 'choices_question':
            if question['options'] == []: 
                warnings.warn(f'question: {question_id} has no options, skip it')
                continue
            choices_text, gold_tag = load_chocies_text_and_gold_tag(question)
            data = [question_id, template_id, knowledge_id, target_view, question_text, choices_text, gold_tag]
            db.insert_into_choices_question(data)
        elif question_type == 'qa_question':
            answer_text = question['object']
            data = [question_id, template_id, knowledge_id, target_view, question_text, answer_text]
            db.insert_into_qa_question(data)
        else:
            raise ValueError(f'question_type: {question_type} not found')
        
    print(f'Storing {question_type} with {target_view} questions to db finished.\n')
        

def main():
    question_dir = 'data/questions/world_knowledge'
    db = EvalDataBase('data/database/world.db')
    question_type = 'qa_question'
    for question_path in Path(question_dir).glob('*.jsonl'):
        print(f'Loading questions from {question_path}...')
        questions = load_questions(question_path)
        target_view = question_path.stem
        store_questions_to_db(questions, target_view, db, question_type)


if __name__ == "__main__":
    main()