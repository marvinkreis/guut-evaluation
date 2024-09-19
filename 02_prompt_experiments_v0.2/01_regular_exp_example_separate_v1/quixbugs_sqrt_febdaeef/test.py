from sqrt import sqrt

def test_correct_sqrt():
    """Testing the correct square root implementation."""
    values = [2, 4, 9]  # Simple test inputs

    for x in values:
        output = sqrt(x, 0.01)
        expected = x ** 0.5
        print(f"Correct sqrt({x}) = {output}, Expected = {expected}")

def test_mutant_sqrt():
    """Testing the mutant square root implementation."""
    def mutant_sqrt(x, epsilon):
        return 100  # Deliberately incorrect output

    values = [2, 4, 9]  # Simple test inputs

    for x in values:
        output = mutant_sqrt(x, 0.01)
        print(f"Mutant sqrt({x}) = {output}, Expected = 100")

# Run both tests
test_correct_sqrt()
test_mutant_sqrt()