from unittest.mock import patch
from types import SimpleNamespace
import sys


def test():
    def mock_init():
        assert False
    sys.modules["colorama"] = SimpleNamespace(init=mock_init)
    with patch("platform.system", return_value="Windows2"):
        import pdir.api

