import flake8.utils as utils
from flake8 import exceptions
from collections import UserString
from unittest.mock import patch
from types import SimpleNamespace


class MockString(UserString):
    pass

class EqToAll():
    def __eq__(self, other):
        return True


def test():
    with patch("inspect.signature", return_value=SimpleNamespace(parameters={"a": SimpleNamespace(kind=[], POSITIONAL_OR_KEYWORD=[], name="", default=[], empty=[])})):
        assert utils.parameters_for(SimpleNamespace(plugin=None))


