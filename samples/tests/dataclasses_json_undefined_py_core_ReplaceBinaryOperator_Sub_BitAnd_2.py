from dataclasses_json import dataclass_json, Undefined, CatchAll
from dataclasses_json.undefined import _CatchAllUndefinedParameters
from dataclasses import dataclass
from typing import Optional, Mapping

@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class A:
    a: int
    catchall: CatchAll

def test():
    try:
        A(1,2,3,4)
    except TypeError:
        assert False
