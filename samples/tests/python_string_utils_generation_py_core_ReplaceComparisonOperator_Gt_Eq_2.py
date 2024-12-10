from string_utils.generation import roman_range

i = 0
class MockInt(int):
    def __eq__(self, other):
        global i
        i += 1
        return super().__eq__(other)

def test():
    try:
        roman_range(MockInt(1), 123, 1)
    except OverflowError:
        pass

    assert i == 0
