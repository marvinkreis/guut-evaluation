from string_utils.manipulation import prettify

def test__prettify():
    """The mutant introduces unwanted spaces before the 's in Saxon genitives."""
    input_string = "Sarah' s book is different from Paul's car, but Tim' s game is fun!"
    output = prettify(input_string)
    
    # Check whether the output is correctly formatted without unwanted spaces.
    assert output == "Sarah's book is different from Paul's car, but Tim's game is fun!", "prettify must not introduce extra spaces in Saxon genitives"