from unittest.mock import patch
from flutes.multiproc import kill_proc_tree


class MockProcess:
    def children(self, recursive: bool):
        assert recursive
        return [self]

    def kill(self):
        pass

    def is_running(self) -> bool:
        return False

    def wait(self, timeout: int):
        pass


def test():
    with patch("psutil.Process", return_value=MockProcess()):
        kill_proc_tree(9999, False)
