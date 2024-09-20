from string_utils.generation import random_string

def test__random_string():
    """The mutant allows random_string to accept size 0, which should raise an error."""
    try:
        output = random_string(0)
        assert False, "Expected ValueError for size 0 but got output: {}".format(output)
    except ValueError as e:
        assert str(e) == 'size must be >= 1', "Unexpected ValueError message: {}".format(e)

    # Additionally test with a valid size to ensure correct behavior
    output = random_string(5)
    assert len(output) == 5, "Expected string of length 5 but got length: {}".format(len(output))