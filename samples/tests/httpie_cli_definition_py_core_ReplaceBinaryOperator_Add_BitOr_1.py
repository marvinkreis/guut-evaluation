from httpie.plugins.registry import plugin_manager
from httpie.plugins.base import AuthPlugin


class A(AuthPlugin):
    description = "Some description"

def test():
    plugin_manager.register(A)

    try:
        from httpie.cli.definition import auth
    except TypeError:
        assert False

