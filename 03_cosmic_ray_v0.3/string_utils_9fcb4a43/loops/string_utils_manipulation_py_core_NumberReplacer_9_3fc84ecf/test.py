from string_utils.manipulation import roman_encode

def test__roman_encode():
    """The mutant changes the hundreds mapping, which leads to an incorrect or error output for certain inputs."""
    output_100 = roman_encode(100)
    assert output_100 == 'C', "Encoding 100 should return 'C'"

    output_300 = roman_encode(300)
    assert output_300 == 'CCC', "Encoding 300 should return 'CCC'"