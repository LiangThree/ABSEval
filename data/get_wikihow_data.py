import requests
from bs4 import BeautifulSoup
from tqdm import *
import nltk
import re
import sqlite3

nltk.download('averaged_perceptron_tagger')

class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def insert_question(self, category, abstract_question, frequency):
        cursor = self.conn.cursor()
        query = '''INSERT INTO abstract_question (category, abstract_question, frequency) VALUES (?, ?, ?)'''
        cursor.execute(query, (category, abstract_question, frequency))
        self.conn.commit()


def check_part_of_speedch(text):
    # text = "What Does DW Mean"
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags


def filter_script(input):
    input = input.lower()
    if '%' in input:
        return False
    check = check_part_of_speedch(input)
    if check[0][1] == 'WRB':
        return input.capitalize()
    if check[0][1] in ['VB']:
        return "How to " + input
    if len(check) < 3:
        return False


def get_sub_category_links(category, sub_category, num_case):
    url = f"https://www.wikihow.com/Category:{sub_category}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    article_links = []
    for link in soup.find_all("a"):
        article_links.append(link.get("href"))

    eval_db = EvalDataBase('data/database/script.db')
    article_links = list(set(article_links))
    # print(f'{sub_category} link count: {len(article_links)}')
    # print(article_links)

    count = 0
    for link in article_links[num_case:3*num_case]:
        if type(link) == str and "https://www.wikihow.com/" in link:
            question = link[24:].replace("-", " ")
            views = get_views(link)
            if views:
                question_after_filter = filter_script(question)
                if question_after_filter:
                    category = category.replace('-', ' ')
                    eval_db.insert_question(category, question_after_filter, views)
                    count += 1
                    # if count >= num_case:
                    #     return count
    return count


def get_wikihow_article_links(category):
    # 目标类别的URL
    url = f"https://www.wikihow.com/Category:{category}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # print(soup)

    article_links = []
    for link in soup.find_all("a"):
        article_links.append(link.get("href"))

    eval_db = EvalDataBase('data/database/script.db')
    article_links = list(set(article_links))

    count = 0
    # for link in tqdm(article_links, desc=f'get link'):
    #     if type(link) == str and "https://www.wikihow.com/" in link:
    #         question = link[24:].replace("-", " ")
    #         views = get_views(link)
    #         if views:
    #             question_after_filter = filter_script(question)
    #             if question_after_filter:
    #                 category = category.replace('-', ' ')
    #                 eval_db.insert_question(category, question_after_filter, views)
    #                 count += 1

    subcategory_div = soup.find_all('div', class_='subcategory')
    sub_category = []
    for one_sub_category in subcategory_div:
        cat_links = one_sub_category.find_all('a', class_='cat_link')
        for one_cat_link in cat_links:
            sub_category.append(one_cat_link.get("href")[10:])

    # print(f"{category}:", len(sub_category))
    # return

    num_case = (300//len(sub_category))+1
    for sub_category in tqdm(sub_category, desc=f'get sub_category'):
        sub_count = get_sub_category_links(category, sub_category, num_case)
        count += sub_count
        if count >= 300:
            return



def get_views(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        views_element = soup.find("div", class_="sp_box sp_stats_box")

        if views_element:
            views = views_element.text.strip()  # 获取"Views"的文本内容
            views_match = re.search(r"Views:\s*(.*)", views)
            if views_match:
                views = views_match.group(1)
                return views

    return False

def update_category_name():
    conn = sqlite3.connect('data/database/script.db')
    cursor = conn.cursor()
    query = '''SELECT * FROM abstract_question'''
    cursor.execute(query, ())
    result = cursor.fetchall()

    for row in tqdm(result):
        question_id, category, question, views = row
        category = category.replace('-', ' ')
        query = '''UPDATE abstract_question SET category = ? WHERE question_id = ?'''
        cursor.execute(query, (category, question_id))
        conn.commit()


if __name__ == '__main__':
    all_category = ["Arts-and-Entertainment", "Computers-and-Electronics" ,"Education-and-Communications",
                "Food-and-Entertaining", "Finance-and-Business", "Health", "Hobbies-and-Crafts",
                "Holidays-and-Traditions", "Home-and-Garden", "Personal-Care-and-Style", "Pets-and-Animals",
                "Philosophy-and-Religion", "Relationships", "Sports-and-Fitness", "Travel"]
    category_list = []

    category_dict = {}
    for category in category_list:
        link = get_wikihow_article_links(category)






