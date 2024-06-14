from llm_eval.data.instance import Instance, Input
from typing import List
from pathlib import Path
import pandas as pd


class WikidataScenario:
    def get_instances(self) -> List[Instance]:
        root = Path('data/questions/2023-10-17-08-10/')
        file_paths = list(root.iterdir())

        questions: List[str] = []
        for file_path in file_paths:
            names = ['obj', 'prop', 'sub', 'question']
            df = pd.read_csv(file_path, names=names, sep='\t')
            for row in df.itertuples():
                questions.append(row.question)
                
        instances: List[Instance] = []
        for question in questions:
            instance = Instance(
                input=Input(question),
                references=[]
            )
            instances.append(instance)
        instances = instances[:3]
        return instances
