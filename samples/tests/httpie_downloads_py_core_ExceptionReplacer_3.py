from httpie.downloads import Downloader
from unittest.mock import MagicMock
import requests


def test():
    downloader = Downloader()
    response = MagicMock(requests.Response)

    try:
        downloader.start("", response)
    except AttributeError:
        pass
    except NameError:
        assert False
