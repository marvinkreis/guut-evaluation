from unittest.mock import patch


count = 0
def mock_overload(fn):
    global count
    count += 1
    return fn

def test():
    with patch("typing.overload", wraps=mock_overload):
        import flutes.multiproc
    assert count == 22
