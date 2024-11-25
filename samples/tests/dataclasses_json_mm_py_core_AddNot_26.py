from dataclasses_json import dataclass_json
from dataclasses import dataclass, field


@dataclass_json
@dataclass
class A:
    value: int = field(default=1)


def test():
    schema = A.schema()
    assert not schema.fields["value"].allow_none
