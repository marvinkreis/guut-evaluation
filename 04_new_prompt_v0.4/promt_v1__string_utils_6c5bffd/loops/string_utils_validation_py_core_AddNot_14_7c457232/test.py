from string_utils.validation import is_credit_card

def test__is_credit_card_kill_mutant():
    """
    Test the validation of credit card types to distinguish between the baseline and mutant.
    This test checks for valid and invalid card types by asserting the expected outcomes.
    """
    valid_card_mastercard = '5555555555554444'  # Valid Mastercard number
    invalid_card_type = '1234567890123456'      # Invalid card type (not matching any known types)

    # Check for valid Mastercard
    assert is_credit_card(valid_card_mastercard, card_type='MASTERCARD') == True

    # Check for invalid card type and expect KeyError
    try:
        is_credit_card(invalid_card_type, card_type='INVALID_TYPE')
    except KeyError:
        print("KeyError raised as expected for invalid card type.")
    else:
        assert False, "Expected KeyError was not raised."