from string_utils.generation import roman_range

i = 0
class MockInt(int):
    def __eq__(self, other):
        global i
        i += 1
        return super().__eq__(other)

def test():
    orig_isinstance = __builtins__["isinstance"]

    def mock_isinstance(obj, cls):
        if orig_isinstance(obj, cls):
            return True
        return orig_isinstance(obj, MockInt)

    __builtins__["isinstance"] = mock_isinstance
    try:
        roman_range(MockInt(1), 123, 1)
    except OverflowError:
        pass

    assert i == 0
