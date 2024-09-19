from sqrt import sqrt

def test__sqrt():
    """Modifying the convergence condition would cause the mutant to potentially return incorrect results."""
    output = sqrt(10, 0.01)
    expected_output = 3.162319422150883  # Known good approximation
    assert abs(output - expected_output) < 0.01, "squared root approximate not within tolerance"