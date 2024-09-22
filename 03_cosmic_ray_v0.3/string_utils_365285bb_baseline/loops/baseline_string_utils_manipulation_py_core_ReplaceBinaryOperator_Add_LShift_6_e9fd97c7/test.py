from string_utils.manipulation import prettify

def test_prettify_uppercase_after_sign():
    try:
        # This input should trigger the __uppercase_first_letter_after_sign
        # If the original code is working, it should capitalize the 'b'
        result = prettify('foo! bar baz')
        
        # Expecting that 'b' in 'bar' is capitalized
        assert result == 'Foo! Bar baz', f"Expected 'Foo! Bar baz' but got '{result}'"
        
        # If we had the mutant, it would cause an IndexError when trying to access match[2]
        # Thus, we don't need to check for an exception in the original code.
        return  # If we reach here, the original implementation is correct
    except IndexError:
        # If we hit an IndexError here, it means we're likely dealing with the mutant
        assert False, "Mutant detected: raised IndexError"

# To run the test function
test_prettify_uppercase_after_sign()