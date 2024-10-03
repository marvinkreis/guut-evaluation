from string_utils.validation import is_credit_card

def test__credit_card_invalid_cases():
    """
    Test invalid credit card numbers to ensure that the is_credit_card function identifies them correctly.
    The mutant is expected to return True for some invalid credit card scenarios, while the baseline returns False.
    """
    invalid_cards = [
        '1234567890123456',  # Invalid credit card number
        '',  # Empty string
        'not_a_card'  # Invalid non-numeric string
    ]
    
    for card in invalid_cards:
        output = is_credit_card(card)
        print(f"Testing card: {card} -> Output: {output}")
        assert output == False  # Expecting False for all invalid cases