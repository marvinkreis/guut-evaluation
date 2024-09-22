from string_utils.validation import is_integer

def test_is_integer():
    # Valid integer inputs
    assert is_integer('42') == True  # Correctly identifies an integer
    assert is_integer('-7') == True  # Correctly identifies a signed integer
    assert is_integer('0') == True    # Correctly identifies zero as an integer

    # Invalid integer inputs
    assert is_integer('42.0') == False  # Should return False for original and mutant
    assert is_integer('forty two') == False  # Should return False
    assert is_integer('') == False  # Should return False
    assert is_integer('   ') == False  # Should return False
    assert is_integer('3.14') == False  # Should return False
    assert is_integer('1e10.2') == False  # Should return False (invalid)
    assert is_integer('1.0') == False  # Should return False
    assert is_integer('0.01') == False  # Should return False
    assert is_integer('+42') == True  # Edge case, should be true
    assert is_integer('-42') == True  # Edge case, should be true
    assert is_integer('-42.5') == False  # Should return False
    
    print("All assertions passed.")

# To run the test
test_is_integer()