import requests
from unittest.mock import MagicMock
from types import SimpleNamespace

import httpie.downloads as downloads


def test():
    downloads.__dict__["parse_content_range"] = lambda x, y: 1
    def mock_raw_stream(msg, with_headers, with_body, on_body_chunk_downloaded, chunk_size):
        assert chunk_size == 1024 * 8
        raise Exception()
    downloads.__dict__["RawStream"] = mock_raw_stream

    downloader = downloads.Downloader(output_file=SimpleNamespace(
        seek=lambda x: None,
        truncate=lambda: None,
        name="name"
    ), resume=False)

    response = MagicMock(requests.Response)
    response.headers = {"Content-Length": 1}

    try:
        downloader.start("", response)
    except AssertionError:
        raise
    except:
        pass


