import ipdb
from .scenario import Scenario
from llm_eval.data.instance import Instance, Input, Reference, Output, CORRECT_TAG
import sqlite3
import json
import random
from typing import List


class MultiChoiceScenario(Scenario):
    def __init__(self, db_path, table_name, acceleration_method, target_view, seed=2023, train_num=5) -> None:
        super().__init__()
        self.conn = sqlite3.connect(db_path)
        self.table_name = table_name
        self.target_view = target_view
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
        question_id, question_text, choices_text, global_tag = row[0], row[5], row[6], row[7]
        input = Input(question_text)
        references = self.choices_text_to_references(choices_text, global_tag)
        instance = Instance(input=input, references=references, id=question_id)
        return instance
    
    def read_rows(self):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {self.table_name} WHERE target_view = '{self.target_view}'")
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
    
    @staticmethod
    def choices_text_to_references(choices_text, correct_text):
        choices = json.loads(choices_text)
        
        references: List[Reference] = []
        for choice, desc in choices.items():
            output = Output(desc)
            tag = CORRECT_TAG if choice == correct_text else ''
            reference = Reference(output, tag)
            references.append(reference)
            
        return references