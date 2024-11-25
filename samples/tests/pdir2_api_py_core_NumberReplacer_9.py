from pdir.api import PrettyAttribute
from pdir.attr_category import AttrCategory


def dummy_function():
    """1\n2"""
    pass

def test():
    pa = PrettyAttribute("name", (AttrCategory.DESCRIPTOR,), dummy_function)
    assert "\n" not in pa.get_oneline_doc()
