from dataclasses_json import dataclass_json, Undefined, CatchAll
from dataclasses_json.undefined import _CatchAllUndefinedParameters
from dataclasses import dataclass, field
from typing import Optional, Mapping
from copy import deepcopy

class DefaultObject:
    def __init__(self, i: int):
        self.i = i

    def __eq__(self, other):
        return isinstance(other, DefaultObject) and self.i == other.i

@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class A:
    a: int
    catchall: CatchAll

def test():
    result = A.from_dict({"a": 1, "undefined_attribute" : 1})
    assert result.catchall == {"undefined_attribute": 1}

