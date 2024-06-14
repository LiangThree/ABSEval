import json
import sqlite3
import random
import os
from pprint import pprint

import yaml
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from proplot import rc


class EvalDataBase:
    def __init__(self, db_path):
        """
        Initialize the database class with a connection to the SQLite database
        """
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def get_eval_result(self):
        query = """
            SELECT * FROM eval_result
        """
        cursor = self.conn.cursor()
        cursor.execute(query, ())
        result = cursor.fetchall()
        return result



if __name__ == '__main__':
