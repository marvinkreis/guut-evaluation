from string_utils.validation import is_palindrome

def test_is_palindrome():
    # Clearly define different cases
    assert is_palindrome('A') == True  # Single character - palindrome
    assert is_palindrome('AA') == True  # Two identical characters - palindrome
    assert is_palindrome('AAA') == True  # Three identical characters - palindrome
    assert is_palindrome('ABA') == True  # Odd-length palindrome
    assert is_palindrome('AABAA') == True  # Longer odd-length palindrome
    assert is_palindrome('ABBA') == True  # Even-length palindrome
    assert is_palindrome('LOOL') == True  # Even-length palindrome
    assert is_palindrome('RACECAR') == True  # Classic palindrome
    assert is_palindrome('MADAM') == True  # Classic palindrome
    assert is_palindrome('HELLO') == False  # Not a palindrome
    assert is_palindrome('WORLD') == False  # Not a palindrome
    assert is_palindrome('AAB') == False  # Not a palindrome
    assert is_palindrome('AAAB') == False  # Not a palindrome
    assert is_palindrome('XYZ') == False  # Not a palindrome

# Execute the test case
test_is_palindrome()