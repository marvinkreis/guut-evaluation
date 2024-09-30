import re
from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """
    Test whether the regex for 'SPACES_AROUND' correctly matches when there are no spaces around a plus operator.
    The input 'hello+world' will pass the original regex as it identifies that there are no spaces around the '+'.
    The mutant will raise a TypeError from an invalid regex due to using '/' instead of '|', which will cause this test to fail with the mutant.
    """
    # This input should match the pattern intended by the original regex
    output = PRETTIFY_RE['SPACES_AROUND'].search('hello+world')  # No spaces
    
    # The baseline regex should successfully match this string
    assert output is not None, "The match should succeed in the baseline code."

# A dummy invokable to run the test
if __name__ == "__main__":
    test__PRETTIFY_RE()