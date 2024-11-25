from httpie.config import BaseConfigDict, ConfigFileError
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace


class MockErrno:
    def __eq__(self, other):
        return True


def test():
    import errno
    orig_errno = errno.ENOENT
    errno.ENOENT = MockErrno()

    with TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "whatever"
        bcd = BaseConfigDict(path)

        try:
            bcd.load()
        except ConfigFileError:
            assert False
