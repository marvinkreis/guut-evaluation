import flake8.utils as utils
from flake8 import exceptions


def test():
    utils.__dict__["_tokenize_files_to_codes_mapping"] = lambda x: [utils._Token("zzz", "")]

    try:
        utils.parse_files_to_codes_mapping("   project/__init__.py:F401 setup.py:E121")
        assert False
    except exceptions.ExecutionError:
        pass


