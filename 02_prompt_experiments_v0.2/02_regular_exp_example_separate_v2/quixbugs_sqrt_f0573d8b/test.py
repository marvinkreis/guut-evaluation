from sqrt import sqrt

def test__sqrt():
    # The correct implementation should return a value close to 2.0 for sqrt(4, 0.01)
    output = sqrt(4, 0.01)
    print(f"output = {output}")
    # Verification that the output is within the desired approximation
    assert abs(output - 2.0) < 0.01, "sqrt must approximate the square root of 4 within the specified epsilon"