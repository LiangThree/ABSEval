from llm_eval.data.instance import Instance
from llm_eval.data.request import Request, Prompt
from .adapter import Adapter
from typing import List


class GenerationAdapter(Adapter):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'generation_adapter'
    
    def generate_request(self, instance: Instance) -> Request:
        prompt = Prompt(instance=instance, text=instance.input.text, adapter_name=self.name)
        return Request(
            question_type='qa',
            instance=instance,
            prompt=prompt
        )

    def adapt(self, instances: List[Instance]) -> List[Request]:
        return [self.generate_request(instance) for instance in instances]
