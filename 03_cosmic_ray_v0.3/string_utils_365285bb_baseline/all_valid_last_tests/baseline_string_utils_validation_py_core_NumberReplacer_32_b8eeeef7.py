from string_utils.validation import is_email

def test_is_email():
    # Valid emails
    assert is_email('my.email@the-provider.com') == True  # Valid email
    assert is_email('user.name+tag+sorting@example.com') == True  # Valid email
    assert is_email('example@domain.com') == True  # Valid email

    # Invalid email tests
    assert is_email('plainaddress') == False  # Invalid format (missing '@')
    assert is_email('@missinglocal.com') == False  # Missing local part
    assert is_email('user@.com') == False  # Invalid domain
    assert is_email('user@domain..com') == False  # Invalid (double dot)
    assert is_email('@domain.com') == False  # Missing local part
    assert is_email('invalid@domain') == False  # Missing TLD

    # More edge cases that should fail on both implementations
    assert is_email('invalid@ domain.com') == False  # Space should make it invalid
    assert is_email('invalid@domain .com') == False  # Space should make it invalid
    assert is_email('invalid@domain.com ') == False  # Trailing space should invalidate
    assert is_email('invalid@.domain.com') == False  # Invalid address with dot at start

    # Cases that should be valid
    assert is_email('example.name+tag@sub.domain.com') == True  # Complex valid email
    assert is_email('test.email@example.com') == True  # Valid type
    assert is_email('valid_email@domain.com') == True  # Valid non-complex email

    # Edge case for a leading space before a valid email
    assert is_email(' valid.email@domain.com') == False  # Leading space should invalidate

    print("All tests passed!")

# Execute the test
test_is_email()