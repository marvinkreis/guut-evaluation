from apimd.compiler import is_staticmethod
import copy


class MockType(type):
    def __eq__(self, other):
        return True

# s = MockType('MockStaticMethod', (staticmethod,), {})

class MockStaticMethod(staticmethod, metaclass=MockType):
    pass

class TestClass:
    @MockStaticMethod
    def static_method():
        pass

def test():
    assert not is_staticmethod(TestClass, "static_method")
