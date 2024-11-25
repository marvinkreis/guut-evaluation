from httpie.cli.definition import auth

def test():
    mutated_option = [a for a in auth._actions if "-A" in a.option_strings][0]
    assert "builtin" not in mutated_option.help

