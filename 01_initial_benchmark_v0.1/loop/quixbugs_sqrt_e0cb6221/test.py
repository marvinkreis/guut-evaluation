from sqrt import sqrt

def test__sqrt():
    x = 2
    epsilon = 0.01
    output = sqrt(x, epsilon)
    assert (output >= (2 ** 0.5 - epsilon) and output <= (2 ** 0.5 + epsilon)), "sqrt must be within the expected range"