from dataclasses_json import dataclass_json
from dataclasses import dataclass
from typing import Any


@dataclass_json
@dataclass
class A:
    value: 'B'

class B:
    value: 'B'


def test():
    b = B()
    b.value = b
    a = A(b)

    def default(obj):
        return obj.value

    try:
        a.to_json(default=default)
    except ValueError:
        pass
    except RecursionError:
        assert False
