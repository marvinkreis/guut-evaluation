import sys
import io
from unittest.mock import patch
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path

import httpie.output.writer as writer
from httpie.context import Environment


def test():
    stderr = io.StringIO()
    env = Environment()

    def mock_write_stream(stream, outfile, flush=False):
        assert not flush

    writer.__dict__["write_stream"] = mock_write_stream
    import httpie.core as core

    with TemporaryDirectory() as tempdir:
        temp_path = Path(tempdir)
        core.program(SimpleNamespace(
            download=True,
            output_file=(temp_path / "a").open("w"),
            output_file_specified=True,
            download_resume=False,
            headers={},
            session="asdf",
            session_read_only=True,
            url="http://example.org",
            files=[],
            data=None,
            json=None,
            form=None,
            offline=False,
            multipart=False,
            method="GET",
            chunked=False,
            auth=None,
            params={},
            timeout=None,
            cert=None,
            proxy=[],
            verify="no",
            ssl_version="tls1",
            ciphers=[],
            auth_plugin=None,
            debug=False,
            path_as_is=True,
            compress=False,
            output_options={},
            max_headers=100,
            check_status=False
        ), env)
