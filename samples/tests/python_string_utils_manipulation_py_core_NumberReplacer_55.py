from string_utils.manipulation import __StringCompressor
from unittest.mock import patch

from zlib import compress


def test():
    with patch("zlib.compress", wraps=compress) as c:
        __StringCompressor.compress("abcd")
        c.assert_called_with(b"abcd", 9)
