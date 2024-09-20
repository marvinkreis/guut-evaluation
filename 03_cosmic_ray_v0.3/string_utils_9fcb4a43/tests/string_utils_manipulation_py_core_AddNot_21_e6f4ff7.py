from string_utils.manipulation import shuffle

def test__shuffle():
    """The mutant changes the condition and leads to an error for valid inputs."""
    output = shuffle("hello")
    assert len(output) > 0, "shuffle must return a non-empty string"