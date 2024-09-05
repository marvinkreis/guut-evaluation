from get_factors import get_factors

def test_get_factors():
    # Test with a composite number (should return its prime factors)
    assert get_factors(12) == [2, 2, 3]  # Factors of 12 are 2 * 2 * 3
    
    # Test with a prime number (should return the number itself)
    assert get_factors(13) == [13]  # Factors of 13 are just 13 itself
    
    # Test with a known composite number (should return [2, 2, 5, 5])
    assert get_factors(100) == [2, 2, 5, 5]  # Factors of 100 are 2 * 2 * 5 * 5
    
    # Repeat the test to ensure consistency
    assert get_factors(101) == [101]  # Factors of 101 (a prime number)