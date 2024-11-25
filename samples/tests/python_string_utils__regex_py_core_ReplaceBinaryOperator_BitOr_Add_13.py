import re


class MockReFlag:
    def __init__(self, orig):
        self.orig = orig
    def __add__(self, other):
        assert False
    def __or__(self, other):
        return self.orig | other
    def __and__(self, other):
        return self.orig & other

def test():
    orig_or = re.RegexFlag.__or__
    def mock_or(self, other):
        if hasattr(other, "orig"):
            return self | other.orig
        else:
            return orig_or(self, other)
    re.RegexFlag.__or__ = mock_or

    re.MULTILINE = MockReFlag(re.MULTILINE)
    re.IGNORECASE = MockReFlag(re.IGNORECASE)
    from string_utils import _regex
