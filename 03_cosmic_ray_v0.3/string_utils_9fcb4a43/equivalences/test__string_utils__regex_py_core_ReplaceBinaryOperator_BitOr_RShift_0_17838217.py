from string_utils import is_json

def test():
    assert is_json('{"a": 1,\n "b": 2}')
