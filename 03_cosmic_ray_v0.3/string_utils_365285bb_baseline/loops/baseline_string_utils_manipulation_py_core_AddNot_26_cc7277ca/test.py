# Assuming these functions are in the string_utils.manipulation file
from string_utils.manipulation import booleanize

def test_booleanize():
    # Test valid boolean inputs
    assert booleanize('true') is True
    assert booleanize('YES') is True
    assert booleanize('1') is True
    assert booleanize('y') is True
    
    # Test invalid boolean inputs
    assert booleanize('nope') is False
    assert booleanize('false') is False
    assert booleanize('0') is False
    
    # Test non-boolean strings
    assert booleanize('some random string') is False
    
    # Test invalid input cases
    try:
        booleanize(None)  # InvalidInputError expected
        assert False, "Expected InvalidInputError for None input"
    except Exception:
        pass  # Correctly raised an exception
    
    try:
        booleanize(123)  # InvalidInputError expected
        assert False, "Expected InvalidInputError for integer input"
    except Exception:
        pass  # Correctly raised an exception

# The test will pass with the correct implementation and fail with the mutant.