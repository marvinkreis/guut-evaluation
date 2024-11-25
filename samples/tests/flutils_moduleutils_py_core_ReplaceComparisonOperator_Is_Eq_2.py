from flutils.moduleutils import _validate_attr_identifier
from collections import UserString


class MockBool:
    def __eq__(self, other):
        assert False

class MockIdentifier(UserString):
    def isidentifier(self):
        print("isid called")
        return MockBool()

def test():
    print(_validate_attr_identifier(MockIdentifier("asdf"), ""))
