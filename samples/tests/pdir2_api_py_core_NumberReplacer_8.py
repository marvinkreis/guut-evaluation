from pdir.api import PrettyAttribute
from pdir.attr_category import AttrCategory

from collections import UserString
from unittest.mock import patch


class MockString(UserString):
    __get__ = 1

    """some doc"""
    def split(self, sep, num):
        assert num == 1
        return super().split(sep, num)

def test():
    with patch("inspect.getdoc", return_value=MockString("xxx")):
        pa = PrettyAttribute("name", (AttrCategory.DESCRIPTOR,), MockString)
        print(pa.get_oneline_doc())
