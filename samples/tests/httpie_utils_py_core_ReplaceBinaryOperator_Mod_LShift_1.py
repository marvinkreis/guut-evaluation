from httpie.utils import get_content_type


def test():
    try:
        get_content_type("some.json.gz")
    except TypeError:
        assert False
