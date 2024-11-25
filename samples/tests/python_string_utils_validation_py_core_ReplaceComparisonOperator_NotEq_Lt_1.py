from string_utils.validation import is_palindrome
import string_utils.validation as validation
from collections import UserString


i = 1
class MockString(UserString):
    def __lt__(self, other):
        global i
        i += 1
        return super().__lt__(other)
    def __getitem__(self, other):
        return MockString(super().__getitem__(other))

def test():
    validation.is_full_string = lambda x: True
    is_palindrome(MockString("abc"))
    assert i > 1
