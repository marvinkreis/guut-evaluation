import sys
from apimd.compiler import load_root
from importlib.machinery import SourceFileLoader
from tempfile import TemporaryDirectory
from pathlib import Path

def test():
    with TemporaryDirectory() as tempdir:
        (Path(tempdir) / "path").mkdir()
        (Path(tempdir) / "path" / "sub1.pyi").write_text("__all__ = []; import sys; sys.modules['goal'] = []")
        (Path(tempdir) / "path" / "sub2.pyi").write_text("__all__ = []; import goal")
        (Path(tempdir) / "path" / "sub3.pyi").write_text("__all__ = []; import goal")
        (Path(tempdir) / "main.py").write_text(f"__path__ = ['{tempdir}/path'];")

        sys.path.append(tempdir)

        try:
            load_root("main", "main")
        except ModuleNotFoundError:
            assert False
