import ipdb
from .scenario import Scenario
from llm_eval.data.instance import Instance, Input, Reference, Output, CORRECT_TAG
import sqlite3
import json
import random
from typing import List


class QuestionAnsweringScenario(Scenario):
    def __init__(self, db_path, table_name, category, acceleration_method, seed=2023, train_num=5) -> None:
        super().__init__()
        self.conn = sqlite3.connect(db_path)
        self.table_name = table_name
        self.category = category
        self.seed = seed
        self.train_num = train_num
        self.acceleration_method = acceleration_method
        
    def get_instances(self):
        # read rows from database
        rows = self.read_rows()
        
        # convert rows to instances
        instances = []
        for row in rows:
            instance = self.row_to_instance(row)
            instances.append(instance)
            
        # split train and test
        self.train_test_split(instances)
        
        return instances
    
    def row_to_instance(self, row):
        question_id, question_text, answer_text = row[0], row[4], row[5]
        input = Input(question_text)
        references = [Reference(Output(answer_text), CORRECT_TAG)]
        instance = Instance(input=input, references=references, id=question_id)
        return instance
    
    def read_rows(self):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {self.table_name} WHERE category = '{self.category}'")
        rows = cursor.fetchall()
        return rows
    
    def train_test_split(self, instances):
        random.seed(self.seed)
        random.shuffle(instances)
        for i, instance in enumerate(instances):
            if i < self.train_num:
                instance.split = 'train'
            else:
                instance.split = 'test'
