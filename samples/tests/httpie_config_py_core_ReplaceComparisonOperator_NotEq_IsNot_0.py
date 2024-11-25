from httpie.config import BaseConfigDict
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace


class MockErrno:
    def __eq__(self, other):
        return True


def test():
    import errno
    orig_errno = errno.EEXIST
    errno.EEXIST = MockErrno()

    with TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "whatever"
        bcd = BaseConfigDict(path)
        bcd.ensure_directory()
