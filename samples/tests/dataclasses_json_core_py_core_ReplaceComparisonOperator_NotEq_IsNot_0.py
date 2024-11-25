from dataclasses_json import dataclass_json
from dataclasses_json.core import _decode_dataclass
import dataclasses_json.core as core
from dataclasses import dataclass
from typing import Collection



class MockType(type):
    def __eq__(self, other):
        return True
    def __hash__(self):
        return 1

class MockString(metaclass=MockType):
    __args__ = []

@dataclass_json
@dataclass
class A:
    value: MockString


def test():
    core._is_collection = lambda x: True
    try:
        _decode_dataclass(A, dict(value="value"), True)
    except IndexError:
        assert False
