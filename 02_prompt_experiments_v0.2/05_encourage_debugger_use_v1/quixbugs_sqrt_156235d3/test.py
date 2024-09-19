from sqrt import sqrt

def test__sqrt():
    """The mutant changes the convergence criteria, causing incorrect square root approximations."""
    expected_sqrt = 1.414213562373095  # This is sqrt(2)
    epsilon = 0.01
    output = sqrt(2, epsilon)
    
    # Check if the output is close to the expected value
    assert expected_sqrt - epsilon <= output <= expected_sqrt + epsilon, "sqrt must be within the expected range"