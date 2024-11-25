from dataclasses_json import dataclass_json, Undefined, CatchAll
from dataclasses_json.undefined import _CatchAllUndefinedParameters
from dataclasses import dataclass, field
from typing import Optional, Mapping
from copy import deepcopy


@dataclass_json(undefined=Undefined.INCLUDE)
@dataclass
class A:
    a: int
    catchall: CatchAll = field(default_factory=lambda: 1)

def test():
    orig_len = len
    def patched_len(l):
        try:
            if l["undefined_attribute"] == 1:
                __builtins__["len"] = orig_len
                return -1
        except Exception:
            return orig_len(l)
        return orig_len(l)
    __builtins__["len"] = patched_len

    result = A.from_dict({"a": 1, "undefined_attribute" : 1, "catchall": 1})
    assert result.catchall == {"undefined_attribute": 1}

