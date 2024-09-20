from string_utils.validation import is_credit_card

def test__is_credit_card():
    """The mutant will always return True for any input, while the correct code checks validity."""
    invalid_card = ""
    valid_card = "4111111111111111"  # Valid credit card for testing

    # Check that an invalid card returns False
    assert not is_credit_card(invalid_card), "An empty string should not be considered a valid credit card"
    # Check that a valid card returns True
    assert is_credit_card(valid_card), "A valid card should be considered valid"