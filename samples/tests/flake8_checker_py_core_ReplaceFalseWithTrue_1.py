import flake8.checker as checker
from types import SimpleNamespace

def do_imap(*args, **kwargs):
    raise Exception()

terminated = False
def do_terminate():
    global terminated
    terminated = True

def test():
    checker.Manager._job_count = lambda self: 1
    manager = checker.Manager(SimpleNamespace(options=SimpleNamespace(exclude=[], extend_exclude=[])), [], [])
    checker._try_initialize_processpool = lambda jobs: SimpleNamespace(imap_unordered=do_imap, terminate=do_terminate)
    try:
        manager.run_parallel()
    except Exception:
        pass
    assert terminated
