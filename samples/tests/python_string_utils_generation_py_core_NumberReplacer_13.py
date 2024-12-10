from string_utils.generation import roman_range


class MockInt(int):
    def __gt__(self, other):
        assert other != -1
        return super().__gt__(other)

def test():
    roman_range(2, 1, MockInt(1))
