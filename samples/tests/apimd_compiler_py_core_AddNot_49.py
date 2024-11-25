import sys
from apimd.compiler import load_root
from importlib.machinery import SourceFileLoader
from tempfile import TemporaryDirectory
from pathlib import Path

def test():
    with TemporaryDirectory() as tempdir:
        (Path(tempdir) / "path").mkdir()
        (Path(tempdir) / "path" / "sub.py").write_text("__all__ = []")
        (Path(tempdir) / "path" / "sub.pyi").touch()
        (Path(tempdir) / "main.py").write_text(f"__path__ = ['{tempdir}/path']; import path.sub")

        sys.path.append(tempdir)
        assert "sub" in load_root("main", "main")
