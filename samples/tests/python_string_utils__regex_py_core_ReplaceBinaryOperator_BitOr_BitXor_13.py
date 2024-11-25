import re

re_or = re.RegexFlag.__or__
re_and = re.RegexFlag.__and__

class MockReFlag:
    def __init__(self, orig):
        self.orig = orig
    def __add__(self, other):
        return re_or(getattr(self, "orig", self), getattr(other, "orig", other))
    def __or__(self, other):
        return re_or(getattr(self, "orig", self), getattr(other, "orig", other))
    def __and__(self, other):
        return re_and(getattr(self, "orig", self), getattr(other, "orig", other))
    def __xor__(self, other):
        assert False
        return re_or(getattr(self, "orig", self), getattr(other, "orig", other))

def test():
    re.RegexFlag.__or__ = MockReFlag.__or__
    re.RegexFlag.__and__ = MockReFlag.__and__
    re.RegexFlag.__xor__ = MockReFlag.__xor__
    from string_utils import _regex
