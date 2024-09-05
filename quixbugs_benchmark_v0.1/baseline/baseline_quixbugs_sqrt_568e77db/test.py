from sqrt import sqrt

def test__sqrt():
    # Test for a known square root with a specific epsilon
    x = 4
    epsilon = 0.01
    result = sqrt(x, epsilon)
    
    # The expected correct result for sqrt(4) is 2
    expected = 2
    
    # Check if the absolute difference is less than epsilon
    assert abs(result - expected) < epsilon, f"Expected a result close to {expected}, got {result}"