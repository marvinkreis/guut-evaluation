from collections import UserDict
from types import SimpleNamespace

import httpie.cli.argparser as argparser


class MockDict(UserDict):
    def __bool__(self):
        return True

def test():
    argparser.__dict__["RequestItems"] = SimpleNamespace(
            from_args=lambda request_item_args, as_form: SimpleNamespace(
                headers=[],
                data=[],
                files=MockDict(),
                params=[],
                multipart_data=[]
    ))

    parser = argparser.HTTPieArgumentParser()
    parser.args = SimpleNamespace(request_items=[], form=False)

    try:
        parser._parse_items()
    except KeyError:
        assert False
    except Exception:
        pass
