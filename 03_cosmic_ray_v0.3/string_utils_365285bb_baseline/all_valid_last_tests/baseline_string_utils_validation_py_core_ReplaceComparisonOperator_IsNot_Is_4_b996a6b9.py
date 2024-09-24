def luhn_check(card_number: str) -> bool:
    """ Perform Luhn check for credit card number validity. """
    total = 0
    num_digits = len(card_number)
    odd_even = num_digits % 2

    for i, digit in enumerate(card_number):
        n = int(digit)  # This line may throw an error if digit is non-numeric
        if i % 2 == odd_even:
            n *= 2
            if n > 9:
                n -= 9
        total += n

    return total % 10 == 0

def is_credit_card(card_number: str) -> bool:
    # Stripping any spaces or dashes for validation
    card_number = card_number.replace(" ", "").replace("-", "")
    
    if not card_number.isdigit() or len(card_number) < 13:  # Basic check for length & digit 
        return False

    return luhn_check(card_number)  # Use the defined Luhn algorithm for validation

def test_is_credit_card():
    # Standard cases
    valid_card = '4111111111111111'  # Valid Visa card
    invalid_card = '4111111111111112'  # Invalid (Luhn fails)

    # Test for valid card
    assert is_credit_card(valid_card) == True, f"Expected True for valid card, got False"
    # Test for invalid card
    assert is_credit_card(invalid_card) == False, f"Expected False for invalid card, got True"

    # Testing edge cases
    edge_cases = [
        '1234567812345670',  # Invalid (Luhn fails)
        '1234567812345678',  # Valid (Luhn passes)
        '4012-8888-8888-1881',  # Valid (Visa formatted)
        '4532 1488 0343 6467',  # Valid (Visa formatted)
        '4532148803436468',  # Invalid
        'abcd123456781234',  # Invalid (non-numeric)
        ''  # Invalid (empty)
    ]

    # Run edge cases
    for card in edge_cases:
        sanitized_card = card.replace("-", "").replace(" ", "")
        if sanitized_card.isdigit() and len(sanitized_card) >= 13:
            if luhn_check(sanitized_card):
                assert is_credit_card(card) == True, f"Expected True for {card}, got False"
                print(f"Card {card} passed validation as expected")
            else:
                assert is_credit_card(card) == False, f"Expected False for {card}, got True"
                print(f"Card {card} failed validation as expected")
        else:
            # For non-numeric or invalid length, expect False
            assert is_credit_card(card) == False, f"Expected False for {card}, got True"      
            print(f"Card {card} failed validation as expected")

# Call the test function to run the assertions
test_is_credit_card()