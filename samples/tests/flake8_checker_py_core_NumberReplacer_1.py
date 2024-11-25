import multiprocessing
import sys
from flake8.checker import _multiprocessing_is_fork

def test():
    multiprocessing.set_start_method("spawn")
    sys.version_info = (3, 3)
    assert _multiprocessing_is_fork()
