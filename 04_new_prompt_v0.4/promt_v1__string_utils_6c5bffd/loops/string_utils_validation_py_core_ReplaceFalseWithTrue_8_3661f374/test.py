from string_utils.validation import is_snake_case

def test__is_snake_case():
    """
    Test the `is_snake_case` function to ensure it correctly identifies valid
    snake_case strings. The mutant incorrectly returns True for empty strings,
    which should return False. This test will confirm the correct behavior.
    """
    test_cases = [
        ("myBlogPost", False),  # Camel case
        ("my blog post", False),  # Contains spaces
        ("my_snake_case", True),  # Valid snake case
        ("my-snake-case", False),  # Invalid separator
        ("", False),  # Empty string
    ]
    
    for input_str, expected in test_cases:
        output = is_snake_case(input_str)
        assert output == expected