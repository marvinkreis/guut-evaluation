from httpie.core import decode_raw_args


class MockType(type):
    def __eq__(self, other):
        return True

# s = MockType('MockBytes', (bytes,), {})

class MockBytes(int, metaclass=MockType):
    pass

def test():
    try:
        decode_raw_args([MockBytes(123)], "utf-8")
    except AttributeError:
        assert False
