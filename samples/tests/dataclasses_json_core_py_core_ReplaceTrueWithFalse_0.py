from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import Any, Union, NewType
from collections import UserDict


@dataclass_json
@dataclass
class B():
    value: int

C = NewType("C", B)

@dataclass_json
@dataclass
class A:
    value: C


import inspect

def test():
    obj = A.from_dict({"value": {"value": 1}})
    assert isinstance(obj.value, B)
