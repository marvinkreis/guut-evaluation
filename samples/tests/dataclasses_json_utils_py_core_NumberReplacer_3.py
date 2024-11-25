from dataclasses_json.utils import _get_type_origin
import sys

class MockType:
    def __init__(self):
        self.__extra__ = "extra"

class MockVersionInfo():
    def __init__(self):
        self.minor = 6

def test():
    real_version_info = sys.version_info
    sys.version_info = MockVersionInfo()
    origin = _get_type_origin(MockType())
    sys.version_info = real_version_info
    assert origin == "extra"

