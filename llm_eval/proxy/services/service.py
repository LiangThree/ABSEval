from abc import ABC, abstractmethod


class Service(ABC):
    @abstractmethod
    def make_request(self, prompt):
        pass
    
    @abstractmethod
    def chat(self, prompt):
        pass
