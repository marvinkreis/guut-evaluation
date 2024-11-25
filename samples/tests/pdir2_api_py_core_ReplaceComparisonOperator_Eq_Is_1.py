from pdir.attr_category import AttrCategory
from pdir.api import PrettyAttribute
from types import SimpleNamespace


def test():
    try:
        pa = PrettyAttribute("name", (int(AttrCategory.DESCRIPTOR),), AttrCategory)
        assert False
    except IndexError:
        pass
