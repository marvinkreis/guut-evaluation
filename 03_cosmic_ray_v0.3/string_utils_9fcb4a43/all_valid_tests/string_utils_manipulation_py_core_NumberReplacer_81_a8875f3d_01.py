from string_utils.manipulation import prettify

def test__prettify():
    """The mutant introduces unwanted spaces before the 's in Saxon genitives."""
    input_string = "John' s pen is different from Jane's and Bob' s book!"
    output = prettify(input_string)
    
    # Check whether the output is correctly formatted without unwanted spaces.
    assert output == "John's pen is different from Jane's and Bob's book!", "prettify must not introduce extra spaces in Saxon genitives"