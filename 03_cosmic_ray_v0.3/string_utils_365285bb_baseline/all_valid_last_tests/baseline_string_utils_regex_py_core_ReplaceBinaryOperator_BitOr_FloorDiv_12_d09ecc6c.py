from string_utils._regex import CREDIT_CARDS

def test__credit_card_regex_detection():
    # A valid VISA credit card number
    valid_visa = "4111111111111111"  # A typical VISA card format
    # An invalid credit card number (MasterCard format)
    invalid_visa = "5111111111111111"  # Begins with 5, not valid for VISA

    # Check if the pattern matches the valid VISA number
    match_valid = CREDIT_CARDS['VISA'].match(valid_visa)
    
    # Assert that we found a match for the valid VISA
    assert match_valid is not None, "Expected to find a valid VISA credit card number, but did not."

    # Check if the pattern does match the invalid VISA number
    match_invalid = CREDIT_CARDS['VISA'].match(invalid_visa)

    # Assert that there is no match for the invalid VISA card number
    assert match_invalid is None, "Unexpected match found for an invalid VISA credit card number."