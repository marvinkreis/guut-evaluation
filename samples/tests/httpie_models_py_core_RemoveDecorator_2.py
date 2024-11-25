from httpie.models import HTTPMessage


def test():
    assert isinstance(HTTPMessage.body, property)
