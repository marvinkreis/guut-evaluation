import logging
import sys
from apimd.compiler import load_root
from importlib.machinery import SourceFileLoader
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from threading import Thread
from io import StringIO


def test():
    output = StringIO()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(output))

    with TemporaryDirectory() as tempdir:
        (Path(tempdir) / "path").mkdir()
        (Path(tempdir) / "path" / "sub1.pyi").write_text("__all__ = []; import missing")
        (Path(tempdir) / "path" / "sub2.pyi").write_text("__all__ = []; import missing")
        (Path(tempdir) / "path" / "sub3.pyi").write_text("__all__ = []; import missing")
        (Path(tempdir) / "main.py").write_text(f"__path__ = ['{tempdir}/path'];")
        sys.path.append(tempdir)

        orig_len = len
        def patched_len(l):
            try:
                if l[0] == 'main.sub1':
                    __builtins__["len"] = orig_len
                    return -1
            except Exception:
                return orig_len(l)
            return orig_len(l)

        __builtins__["len"] = patched_len

        try:
            load_root("main", "main")
        except ModuleNotFoundError:
            pass

        assert "Load stub" not in output.getvalue()
