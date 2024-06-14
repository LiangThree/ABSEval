from dataclasses import dataclass, field
from .wikidata import WikidataItem
from typing import List


@dataclass
class TripleId:
    obj_id: str
    sub_id: str
    claim_id: str


@dataclass
class Qualifier:
    name: str
    value: str


@dataclass
class Triple:
    obj: str
    sub: str
    claim: str
    qualifiers: List[Qualifier] = field(default_factory=list)


@dataclass
class TripleQuestion:
    triple: Triple
    question: str
    source: str
    options: List[str] = None
