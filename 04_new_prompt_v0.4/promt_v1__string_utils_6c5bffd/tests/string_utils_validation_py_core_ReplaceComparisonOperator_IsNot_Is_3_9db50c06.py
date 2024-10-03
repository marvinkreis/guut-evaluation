from string_utils.validation import is_email

def test__is_email():
    """
    Test various email formats to determine if the email validation behaves correctly.
    Specifically, this test checks whether emails with escaped "@" signs are validated 
    correctly. The mutant version alters the logic, which impacts the validation outcome 
    for certain email formats, notably those with multiple "@" symbols.
    """
    test_cases = [
        ('my.email@the-provider.com', True), # Valid email
        ('@gmail.com', False), # Invalid email
        ('my.email@provider.com', True), # Valid email
        ('test\\@example.com', True), # Valid email with escaped "@"
        ('test@example.com', True), # Valid email
        ('test@@example.com', False), # Invalid email (two "@")
    ]
    
    for input_string, expected in test_cases:
        output = is_email(input_string)
        print(f"Input: {input_string}, Output: {output}, Expected: {expected}")
        assert output == expected