from string_utils.manipulation import prettify

def test__prettify():
    # Test case 1: Basic capitalization after a period
    input_string_1 = "hello. what happens now?"
    expected_output_1 = "Hello. What happens now?"
    assert prettify(input_string_1) == expected_output_1, "Test 1 Failed"

    # Test case 2: Ensure capitalize after exclamation
    input_string_2 = "wow! this should capitalize."
    expected_output_2 = "Wow! This should capitalize."
    assert prettify(input_string_2) == expected_output_2, "Test 2 Failed"

    # Test case 3: Incorrectly formatted with no space
    input_string_3 = "hmm...what is next?"
    expected_output_3 = "Hmm... What is next?"
    assert prettify(input_string_3) == expected_output_3, "Test 3 Failed"

    # Test case 4: Commas shouldn't trigger capitalization
    input_string_4 = "hello, how are you?here is my question."
    expected_output_4 = "Hello, how are you? Here is my question."
    assert prettify(input_string_4) == expected_output_4, "Test 4 Failed"

    # Test case 5: Leading spaces to check how well it capitalizes
    input_string_5 = "   congratulations. you did it!"
    expected_output_5 = "Congratulations. You did it!"
    assert prettify(input_string_5) == expected_output_5, "Test 5 Failed"

    # Test case 6: Ensure that trailing punctuation correctly capitalizes the next start
    input_string_6 = "is this working?yes, it is!"
    expected_output_6 = "Is this working? Yes, it is!"
    assert prettify(input_string_6) == expected_output_6, "Test 6 Failed"

    # Test case 7: Testing multiple punctuation with no spaces in between
    input_string_7 = "what?hello!let's go."
    expected_output_7 = "What? Hello! Let's go."
    assert prettify(input_string_7) == expected_output_7, "Test 7 Failed"

    print("All tests passed!")

# Run the test cases
test__prettify()