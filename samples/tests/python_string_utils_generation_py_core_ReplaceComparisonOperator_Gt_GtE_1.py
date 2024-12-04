from string_utils.generation import roman_range


class MockInt(int):
    def __ge__(self, other):
        assert False

def test():
    roman_range(2, 1, MockInt(1))

