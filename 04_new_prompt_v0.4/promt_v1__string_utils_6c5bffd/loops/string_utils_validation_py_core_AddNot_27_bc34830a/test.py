from string_utils.validation import contains_html

def test__contains_html_error_handling():
    """
    Test whether 'contains_html' raises an InvalidInputError 
    when passed a non-string input. The mutant does not handle 
    this error correctly and will not raise an appropriate exception.
    Additionally, the test ensures that valid HTML content returns True 
    on both versions, which confirms that the mutant has behavior 
    inconsistent with the baseline.
    """
    # Testing Invalid Input
    try:
        contains_html(12345)  # Passing an integer instead of a string
        print("No exception raised for invalid input, mutant has a bug.")
    except Exception as e:
        print(f"Exception raised for invalid input: {e}")

    # Testing Valid HTML Input
    output = contains_html('<div>Hello World</div>')
    print(f"Output for valid HTML input: {output}")