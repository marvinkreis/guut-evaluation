import requests
from unittest.mock import MagicMock
from types import SimpleNamespace

import httpie.downloads as downloads


def test():
    downloader = downloads.Downloader(output_file=SimpleNamespace(
        seek=lambda x: None,
        truncate=lambda: None,
        name="name"
    ), resume=False)

    downloads.__dict__["parse_content_range"] = lambda x, y: 1

    def mock_started(resumed_from, total_size):
        assert resumed_from == 0
        raise Exception()

    downloader.status.started = mock_started

    response = MagicMock(requests.Response)
    response.headers = {"Content-Length": 1}

    try:
        downloader.start("", response)
    except AssertionError:
        raise
    except:
        pass

