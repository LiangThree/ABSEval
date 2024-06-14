import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import sqlite3
import ipdb


def load_models(conn):
    cursor = conn.cursor()
    cursor.execute('select distinct model_name from eval_result')
    rows = cursor.fetchall()
    models = [row[0] for row in rows]
    models = [str(model) for model in models]
    return models


def load_target_views(conn):
    cursor = conn.cursor()
    cursor.execute('select distinct target_view from choices_question')
    rows = cursor.fetchall()
    target_views = [row[0] for row in rows]
    target_views = [str(target_view) for target_view in target_views]
    return target_views


def get_model_robustness(conn, model_name):
    cursor = conn.cursor()
    cursor.execute('select eval_result from eval_result where model_name=?', (model_name,))
    rows = cursor.fetchall()
    eval_results = [row[0] for row in rows]
    eval_results = [float(eval_result) for eval_result in eval_results]
    stat = sum(eval_results) / len(eval_results)
    return stat


def get_robustness(conn, model_name, target_view):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT eval_result
        FROM eval_result JOIN choices_question
        ON eval_result.question_id = choices_question.question_id
        WHERE eval_result.model_name=? AND choices_question.target_view=?
    """, (model_name, target_view))
    rows = cursor.fetchall()
    eval_results = [row[0] for row in rows]
    eval_results = [float(eval_result) for eval_result in eval_results]
    try:
        stat = sum(eval_results) / len(eval_results)
    except ZeroDivisionError:
        stat = 0.0
    return stat


def get_model_precision(conn, model_name):
    cursor = conn.cursor()
    cursor.execute('select eval_result from eval_result where model_name=?', (model_name,))
    rows = cursor.fetchall()
    eval_results = [row[0] for row in rows]
    eval_results = [float(eval_result) for eval_result in eval_results]
    stat = sum(eval_results) / len(eval_results)
    return stat
    

def main():
    db_path = 'data/database/world.db'
    conn = sqlite3.connect(db_path)
    models = load_models(conn)
    target_views = load_target_views(conn)
    print('| target_view | model | stat |')
    print('| --- | --- | --- |')
    for target_view in target_views:
        for model in models:
            if model == 'qwen/Qwen-14B': continue
            stat = get_robustness(conn, model, target_view)
            # print('target_view: {}, model: {}, stat: {}'.format(target_view, model, stat))
            print('| {} | {} | {} |'.format(target_view, model, stat))
            
    avg_dict = {}
    for target_view in target_views:
        for model in models:
            if model == 'qwen/Qwen-14B': continue
            stat = get_robustness(conn, model, target_view)
            if model not in avg_dict:
                avg_dict[model] = []
            avg_dict[model].append(stat)
    print('avg')
    print('| model | stat |')
    print('| --- | --- |')
    for model in avg_dict:
        stat = sum(avg_dict[model]) / len(avg_dict[model])
        print('| {} | {} |'.format(model, stat))


if __name__ == '__main__':
    main()