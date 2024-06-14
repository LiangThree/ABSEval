import json
from tqdm import *
import openai
import sqlite3
from openai import OpenAI
import httpx
openai.api_key = ""


class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def insert_question(self, category, target_view, limitation, abstract_question_id, abstract_question, question, frequency):
        cursor = self.conn.cursor()
        query = '''INSERT INTO question (category, target_view, limitation, abstract_question_id, abstract_question, question, frequency) VALUES (?, ?, ?, ?, ?, ?, ?)'''
        cursor.execute(query, (category, target_view, limitation, abstract_question_id, abstract_question, question, frequency))
        self.conn.commit()


    def get_unanswered_abstract_questions(self, category):
        # 连接到数据库
        cursor = self.conn.cursor()
        cursor.execute('''
                        SELECT a.question_id, a.category, a.abstract_question, a.frequency, a.valid
                        FROM abstract_question a
                        LEFT JOIN question q ON a.question_id = q.abstract_question_id
                        WHERE q.abstract_question_id IS NULL AND a.category like ? AND a.valid=1
                    ''', (category,))
        unanswered_abstract_questions = cursor.fetchall()

        return unanswered_abstract_questions

    def clean_question_less_limitation(self):
        cursor = self.conn.cursor()
        query = f'''
            SELECT abstract_question_id, count(*)
            FROM question
            GROUP BY abstract_question_id
        '''
        cursor.execute(query, ())
        answer = cursor.fetchall()

        for one_answer in answer:
            abstract_question_id = one_answer[0]
            question_num = one_answer[1]
            if question_num < 3:
                query = f'''
                    DELETE FROM question
                    WHERE abstract_question_id = {abstract_question_id}
                '''
                cursor.execute(query, ())
                self.conn.commit()

    def select_question_num(self, category):
        cursor = self.conn.cursor()
        query = f'''
            SELECT count(distinct abstract_question_id)
            FROM question
            WHERE category like ?
        '''
        cursor.execute(query, (category,))
        answer_num = cursor.fetchone()

        return answer_num


eval_db = EvalDataBase('data/database/script.db')

def chat(prompt):
    client = OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model='gpt-4-turbo',
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    answer = response.choices[0].message.content
    return answer

def chat_online(prompt):
    client = OpenAI(
        base_url="https://api.xty.app/v1",
        api_key=online_api_key,
        http_client=httpx.Client(
            base_url="https://api.xty.app/v1",
            follow_redirects=True,
        ),
    )

    completion = client.chat.completions.create(
        model='gpt-4-turbo',
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    answer = completion.choices[0].message.content
    return answer

def get_one_limitation(question_id, category, abstract_question, frequency):
    prompt = (f"Create possible Specific Goals according to the Abstract Goal, here are some examples\n"
              f"Abstract Goal: Create a Decision Tree \n"
              "{\"Constraint\": \"on Computer\", \"Specific Goal\": \"Create a Decision Tree on a Computer\"} \n"
              "Here is my question: \n"
              f"Abstract Goal: {abstract_question}"
              "Please answer me in json format {\"Constraint\": \"...\", \"Specific Goal\": \"...\"}."
              "Please answer me in json format, add one restriction and just answer one answer.")
    answer = chat_online(prompt)
    # answer = chat(prompt)

    try:
        target_view = 1
        one_limitation_json = json.loads(answer)
        limitation = one_limitation_json["Constraint"]
        question = one_limitation_json["Specific Goal"]
        abstract_question_id = question_id

        eval_db.insert_question(category, target_view, limitation, abstract_question_id, abstract_question, question,
                                frequency)
    except Exception:
        print("Error: ", Exception)
        print(answer)


def get_two_limitation(question_id, category, abstract_question, frequency):
    prompt = (f"Create possible Specific Goals according to the Abstract Goal, here are some examples. I hope you can add two restrictions for me.\n"
              f"Abstract Goal: Say Goodbye in Different Language \n"
              "{\"Constraint\": \"on Computer, to Help You Choose a Holiday Destination\", \"Specific Goal\": \"Create a Decision Tree on Computer to Help You Choose a Holiday Destination\"} \n"
              f"Abstract Goal: {abstract_question}"
              "Please answer me in json format {\"Constraint\": \"Constraint1, Constraint2\", \"Specific Goal\": \"...\"}."
              "Please answer me in json format, add two restrictions and just answer one answer.")

    answer = chat_online(prompt)
    # answer = chat(prompt)

    target_view = 2
    try:
        one_limitation_json = json.loads(answer)
        limitation = one_limitation_json["Constraint"]
        question = one_limitation_json["Specific Goal"]
        abstract_question_id = question_id
        eval_db.insert_question(category, target_view, limitation, abstract_question_id, abstract_question, question,
                                frequency)
    except:
        print("Error: ", Exception)
        print(answer)


def get_three_limitation(question_id, category, abstract_question, frequency):
    prompt = (f"Create possible Specific Goals according to the Abstract Goal, here are some examples. I hope you can add three restrictions for me.\n"
              f"Abstract Goal: Say Goodbye in Different Language \n"
              "{\"Constraint\": \"on Computer, to Help You Choose a Holiday Destination, with 3 Options\", \"Specific Goal\": \"Create a Decision Tree on Computer to Help You Choose a Holiday Destination with 3 options.\"} \n"
              f"Abstract Goal: {abstract_question}"
              "Please answer me in json format {\"Constraint\": \"Constraint1, Constraint2, Constraint3\", \"Specific Goal\": \"...\"}."
              "Please answer me in json format, add three restrictions and just answer one answer.")

    answer = chat_online(prompt)
    # answer = chat(prompt)

    try:
        target_view = 3
        one_limitation_json = json.loads(answer)
        limitation = one_limitation_json["Constraint"]
        question = one_limitation_json["Specific Goal"]
        abstract_question_id = question_id

        eval_db.insert_question(category, target_view, limitation, abstract_question_id, abstract_question, question,
                                frequency)
    except:
        print("Error: ", Exception)
        print(answer)


def get_special_question(category):
    print(f"current category: {category}")

    unanswered_abstract_questions = eval_db.get_unanswered_abstract_questions(category)
    question_num = int(eval_db.select_question_num(category)[0])

    count = 0
    for row in tqdm(unanswered_abstract_questions, desc='make limitation question'):
        if count == 53-question_num:
            break
        question_id, category, abstract_question, frequency, view = row
        try:
            get_one_limitation(question_id, category, abstract_question, frequency)
            get_two_limitation(question_id, category, abstract_question, frequency)
            get_three_limitation(question_id, category, abstract_question, frequency)
            count += 1
        except Exception as e:
            print('error!')
            print(e)
            print(question_id)
            print(row)
            eval_db.clean_question_less_limitation()



def test():
    question = "How to connect a scanner to a computer"
    answer = chat(question)
    print(answer)

    answer = chat_online(question)
    print(answer)


if __name__ == '__main__':
    # question = "How to clean a window air conditioner"
    # get_three_limitation(question)

    # all_category = ["Arts-and-Entertainment", "Computers-and-Electronics", "Education-and-Communications",
    #                 "Food-and-Entertaining", "Health", "Hobbies-and-Crafts",
    #                 "Holidays-and-Traditions", "Home-and-Garden", "Sports-and-Fitness", "Travel"]
    all_category = ["Health", "Hobbies-and-Crafts", "Holidays-and-Traditions", "Home-and-Garden", "Sports-and-Fitness", "Travel"]
    category_list = all_category

    for category in category_list:
        eval_db.clean_question_less_limitation()
        category = category.replace("-", " ")
        get_special_question(category)
