import sys
import io

import httpie.core as core
from httpie.context import Environment


def test():
    core.program = lambda args, env: sys.exit(1)
    stderr = io.StringIO()
    env = Environment()
    env.stderr = stderr
    core.main("-v xxx".split(), env)
    assert stderr.getvalue()
