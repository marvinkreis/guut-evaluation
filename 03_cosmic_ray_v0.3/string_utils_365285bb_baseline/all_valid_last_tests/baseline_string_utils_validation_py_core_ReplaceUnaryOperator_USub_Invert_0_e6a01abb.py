from string_utils.validation import is_email

def test_is_email():
    # Valid email cases
    assert is_email('user@example.com') == True  # Valid basic email
    assert is_email('user.name@sub.domain.com') == True  # Valid with subdomain
    assert is_email('user+tag@domain.com') == True  # Valid with plus symbol
    assert is_email('username@domain.co.in') == True  # Valid with country code
    assert is_email('user@domain.com') == True  # Basic valid email

    # Invalid email cases that should specifically showcase mutant behavior
    assert is_email('plainaddress') == False  # Invalid, no @ symbol
    assert is_email('@missingusername.com') == False  # Invalid, missing username
    assert is_email('username@.com') == False  # Invalid, starts with dot
    assert is_email('username@domain..com') == False  # Invalid, double dot in domain
    assert is_email('user@domain,com') == False  # Invalid, comma instead of dot
    assert is_email('username@domain.com ') == False  # Invalid, space at end
    assert is_email(' username@domain.com') == False  # Invalid, leading space
    assert is_email('my email@domain.com') == False  # Invalid, space in name
    assert is_email('"my email@domain.com"') == False  # Invalid, space within quotes

    # Tests to exploit mutant behavior:
    assert is_email('"user@domain.com') == False  # Invalid due to unmatched quote
    assert is_email('username@domain.com"') == False  # Invalid due to unmatched quote
    assert is_email('user@domain.com"') == False  # Invalid, closing quote without opening
    assert is_email('"user@domain"') == False  # Invalid - missing TLD

    # Edge quotes without proper context
    assert is_email('"username@domain."') == False  # Invalid, domain must end with TLD
    assert is_email('user@"@domain.com') == False  # Invalid, quotes in the wrong place
    assert is_email('user@domain".com') == False  # Invalid, quote misplaced

    # Additional malformed formats aiming to pinpoint issues in mutant logic
    assert is_email('user@domain.com..') == False  # Invalid due to trailing dots
    assert is_email('invalid@domain..com') == False  # Invalid, double dot in domain again
    assert is_email('..invalid@domain.com') == False  # Invalid because it starts with a dot

    print("All tests passed!")

# Execute the test
test_is_email()