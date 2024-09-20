from string_utils.validation import is_camel_case

def test__is_camel_case():
    """The mutant incorrectly handles non-camel case strings and None inputs."""
    # Valid camel case strings
    assert is_camel_case("MyString") == True, "Expected 'MyString' to be camel case."
    assert is_camel_case("myString") == True, "Expected 'myString' to be camel case."
    assert is_camel_case("mixedCASE") == True, "Expected 'mixedCASE' to be camel case."
    assert is_camel_case("CamelCase123") == True, "Expected 'CamelCase123' to be camel case."
    
    # Invalid cases
    assert is_camel_case("") == False, "Expected empty string to not be camel case."
    assert is_camel_case("Not a camel case") == False, "Expected non-camel case string to not be camel case."
    assert is_camel_case(None) == False, "Expected None to not be camel case."