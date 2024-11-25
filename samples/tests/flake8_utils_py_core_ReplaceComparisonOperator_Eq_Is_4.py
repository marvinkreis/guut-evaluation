import flake8.utils as utils
from flake8 import exceptions
from collections import UserString

class MockString(UserString):
    pass


def test():
    utils.__dict__["_tokenize_files_to_codes_mapping"] = lambda x: [
            utils._Token(utils._COLON, ""),
            utils._Token(MockString(utils._FILE), "")
    ]

    try:
        utils.parse_files_to_codes_mapping("   project/__init__.py:F401 setup.py:E121")
    except exceptions.ExecutionError:
        assert False


