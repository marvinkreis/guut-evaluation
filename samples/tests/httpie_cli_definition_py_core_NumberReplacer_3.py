from httpie.cli.definition import output_processing

def test():
    mutated_option = [a for a in output_processing._actions if "-s" in a.option_strings][0]
    assert "        coffee" in mutated_option.help

