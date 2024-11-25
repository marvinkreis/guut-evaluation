from string_utils.generation import roman_range


class MockInt(int):
    def __gt__(self, other):
        assert other != -1
        return super().__gt__(other)

def test():
    orig_isinstance = __builtins__["isinstance"]

    def mock_isinstance(obj, cls):
        if orig_isinstance(obj, cls):
            return True
        return orig_isinstance(obj, MockInt)

    __builtins__["isinstance"] = mock_isinstance
    roman_range(2, 1, MockInt(1))
