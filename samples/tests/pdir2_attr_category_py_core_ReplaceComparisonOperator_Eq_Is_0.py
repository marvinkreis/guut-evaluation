from pdir.attr_category import AttrCategory, category_match


def test():
    assert category_match(int(AttrCategory.SLOT), AttrCategory.SLOT)
