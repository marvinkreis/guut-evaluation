from pdir.attr_category import AttrCategory
from pdir.api import PrettyAttribute
from types import SimpleNamespace


class A:
    @property
    def __doc__(self):
        raise Exception()

def test():
    pa = PrettyAttribute("name", (AttrCategory.SLOT,), A())
    print(pa.get_oneline_doc())
