from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import Any, Union


@dataclass_json
@dataclass
class B:
    b: int

@dataclass_json
@dataclass
class A:
    a: Union[B, None, int]


def test():
    obj = A.from_dict({"a": {"b": 1}})
    assert not isinstance(obj.a, B)
