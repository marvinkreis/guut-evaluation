from dataclasses_json import dataclass_json
from dataclasses import dataclass, field, MISSING


class B:
    def __eq__(self, other):
        if other is MISSING:
            return True
        else:
            return False

default_factory = B()

@dataclass_json
@dataclass
class A:
    value: int = field(default_factory=default_factory)

def test():
    schema = A.schema()
    assert schema.fields["value"].default is default_factory
