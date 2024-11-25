from dataclasses_json import dataclass_json, Undefined, CatchAll
from dataclasses_json.undefined import _CatchAllUndefinedParameters
from dataclasses import dataclass
from typing import Optional, Mapping
from copy import deepcopy

@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class A:
    a: int
    catchall: CatchAll = 1

def test():
    result = A.from_dict({"a": 1, "undefined_attribute" : 1, "catchall": 1})
    assert result.catchall == {"undefined_attribute": 1}

