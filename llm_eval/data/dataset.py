import pdb
import bz2
import json
import warnings
import pathlib
from llm_eval.common.utils import line_to_json


class WikidataDataset:
    def __init__(self, file_path, file_type='json'):
        self.file_path = file_path
        self.file_type = file_type
        self.file = open(file_path, 'r')

    def __len__(self):
        return 6062702

    def __iter__(self):
        if self.file_type == 'zip':
            while True:
                try:
                    line = self.file.readline()
                    yield line_to_json(line)
                except Exception as e:
                    warnings.warn(e)
        else:
            for line in self.file:
                yield json.loads(line)


class WikiChunkDataset:
    def __init__(self, folder):
        self.folder = pathlib.Path(folder)

    def __iter__(self):
        for file_path in self.folder.glob('*'):
            with open(file_path) as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().strip(',').strip(']')
                if line == '[': continue
                try:
                    yield json.loads(line)
                except Exception as e:
                    warnings.warn(e)
    def __len__(self):
        return 101909641


class JsonlDataset:
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path) as f:
            for line in f:
                item = json.loads(line)
                yield item
