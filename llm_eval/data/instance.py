from dataclasses import dataclass
from typing import List, Optional


CORRECT_TAG = 'correct'


@dataclass(frozen=True)
class Input:
    text: str

@dataclass(frozen=True)
class Output:
    text: str


@dataclass
class Reference:
    """
    一个Instance样例的选项，output代表选项内容，tag代表选项正确性
    """
    output: Output
    tag: str = ''

    @property
    def is_correct(self) -> bool:
        return self.tag == CORRECT_TAG


@dataclass
class Instance:
    """
    一个样例，可以是一个问答题，选择题（单选）或判断题
    """
    input: Input
    references: List[Reference]
    split: Optional[str] = None
    id: Optional[str] = None
    answer: str = None

    @property
    def correct_reference(self):
        for reference in self.references:
            if reference.is_correct:
                return reference
        return None
