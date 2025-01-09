from docstring_parser.rest import parse
from unittest.mock import patch
from types import SimpleNamespace

def test():
    with patch("re.finditer", side_effect=[
        [
            SimpleNamespace(group=lambda n: ":name1: desc1"),
            SimpleNamespace(group=lambda n: None),
            SimpleNamespace(group=lambda n: ":name2: desc2"),
        ]
    ]):
        assert len(parse(":name: val").meta) == 2
