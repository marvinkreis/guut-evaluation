from apimd.compiler import get_name


class MockObject:
     def __getattribute__(self, attr):
         if attr == "__name__":
             return "name"
         elif attr == "__module__":
             return "aaaa"
         elif attr == "__class__":
             raise AttributeError()

def test():
    o = MockObject()
    assert get_name(o) == "aaaa.name"
