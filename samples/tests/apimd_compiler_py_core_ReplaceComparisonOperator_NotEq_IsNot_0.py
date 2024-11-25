import re
from collections import UserString

import apimd.compiler as compiler


class MockString(UserString):
    pass

get_name = compiler.__dict__["get_name"]
def mock_get_name(x):
    return MockString(get_name(x))


def test():
    compiler.__dict__["get_name"] = mock_get_name
    list(compiler.local_vars(compiler))
    assert not compiler.ALIAS
