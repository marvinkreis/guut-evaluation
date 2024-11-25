import logging
import re
import sys
from apimd.compiler import load_root
from importlib.machinery import SourceFileLoader
from io import StringIO
from tempfile import TemporaryDirectory
from pathlib import Path

def test():
    output = StringIO()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(output))

    with TemporaryDirectory() as tempdir:
        (Path(tempdir) / "path").mkdir()
        (Path(tempdir) / "path" / "sub1.pyi").write_text("__all__ = []; import missing;")
        (Path(tempdir) / "path" / "sub2.pyi").write_text("__all__ = []; import missing;")
        (Path(tempdir) / "path" / "sub3.pyi").write_text("__all__ = [];")
        (Path(tempdir) / "main.py").write_text(f"__path__ = ['{tempdir}/path'];")

        sys.path.append(tempdir)

        try:
            load_root("main", "main")
        except ModuleNotFoundError:
            pass

        assert len([match for match in re.finditer("Load stub:", output.getvalue())]) == 3
