import sqlite3


class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def select_inference_from_llm_inference(self, question_id):
        query = """
            SELECT question_id, model_name, inference
            FROM llm_inference
            WHERE question_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (question_id,))
        result = cursor.fetchall()
        return result

    def select_abstract_question(self):
        query = """
            SELECT question_id
            FROM abstract_question
            WHERE EXISTS(
                SELECT 1
                FROM question
                WHERE question.abstract_question_id = abstract_question.question_id
            )
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result

    def select_question(self, abs_question_id):
        query = """
            SELECT *
            FROM question
            WHERE abstract_question_id = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (abs_question_id,))
        result = cursor.fetchall()
        return result


def create_learner_prompt(question, completion) -> str:
    with open('data/metrics/learner_prompt.txt', encoding='utf-8') as f:
        learner_prompt_format = f.read()
    prompt = learner_prompt_format.replace('Question', question)
    examples = ''
    for index, answer in enumerate(completion):
        examples += f'Example {index + 1}:\n'
        examples += f'{answer}\n\n'
    prompt = prompt.replace('EXAMPLES', examples)

    return prompt


def create_execute_prompt(question, limitation, inference) -> str:
    with open('data/metrics/execute_prompt.txt', encoding='utf-8') as f:
        eval_prompt_format = f.read()
    current_prompt = eval_prompt_format.replace('QUESTION', question)
    current_prompt = current_prompt.replace('LIMITATION', limitation)
    current_prompt = current_prompt.replace('MODEL_INFERENCE', inference)

    return current_prompt


def create_eval_prompt(question, gold, inference) -> str:
    with open('data/metrics/eval_prompt.txt', encoding='utf-8') as f:
        eval_prompt_format = f.read()
    current_prompt = eval_prompt_format.replace('Question', question)
    current_prompt = current_prompt.replace('Gold Answer', gold)
    current_prompt = current_prompt.replace('Model Answer', inference)

    return current_prompt


if __name__ == '__main__':
    db = EvalDataBase('data/database/script.db')
    abstract_question = db.select_abstract_question()
    question_dictionary = {}

    for one_abs_question in abstract_question:
        abs_question_id = one_abs_question[0]
        questions = db.select_question(abs_question_id)
        prompt_sum = 0
        for question in questions:
            question_id, _, _, limitation, _, _, question, _, _ = question
            inference = db.select_inference_from_llm_inference(question_id)
            inference = [one[2] for one in inference]
            prompt = create_learner_prompt(question, inference)
            prompt_sum += len(prompt)

        if prompt_sum != 0:
            question_dictionary[abs_question_id] = prompt_sum

    # 按值排序字典
    sorted_dict = sorted(question_dictionary.items(), key=lambda x: x[1])
    # 选择排序后的前 40 个值
    # top_40 = dict(sorted_dict[:40])
    sorted_dict = dict(sorted_dict)
    total_sum = sum(sorted_dict.values())
    gpt_4_price = total_sum * 0.001 * 0.03
    gpt_3_price = total_sum * 0.001 * 0.003

    prompt_dict = {}
    prompt_sum = 0
    execute_count = 0
    eval_count = 0
    for abs_question_id in sorted_dict.keys():
        questions = db.select_question(abs_question_id)
        for question in questions:
            question_id, _, _, limitation, _, _, question, _, _ = question
            inference = db.select_inference_from_llm_inference(question_id)
            inference = [one[2] for one in inference]
            for one_inference in inference:
                execute_prompt = create_execute_prompt(question, limitation, one_inference)
                eval_prompt = create_eval_prompt(question, one_inference, one_inference)
                execute_count += 1
                eval_count += 1
                prompt_sum += len(execute_prompt)
                prompt_sum += len(eval_prompt)
                # print(execute_prompt, eval_prompt)
                # print(prompt_sum)

    gpt_4_price += prompt_sum * 0.001 * 0.03
    gpt_3_price += prompt_sum * 0.001 * 0.003


    eval_output = "{\"missing_steps\":\"True\",\"redundant_steps\":\"True\",\"duplicate_steps\": \"True\",}"
    execute_output = "{\"missing_steps\":\"False\",\"redundant_steps\":\"True\",\"duplicate_steps\": \"False\",}"
    output_sum = len(eval_output)*eval_count + len(execute_output)*execute_count

    gpt_4_price += output_sum * 0.001 * 0.06
    gpt_3_price += output_sum * 0.001 * 0.006

    gpt_4_price = round(gpt_4_price, 2)*10
    gpt_3_price = round(gpt_3_price, 2) * 10

    print('4', gpt_4_price, '$')
    print('3', gpt_3_price, '$')

