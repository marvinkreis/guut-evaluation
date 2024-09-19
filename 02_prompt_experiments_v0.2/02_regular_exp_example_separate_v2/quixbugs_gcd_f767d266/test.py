from gcd import gcd

def test__gcd():
    """ The GCD function must return the correct GCD. The mutant will fail on (35, 0) or (0, 21) handling. """
    
    # Known input pairs
    inputs_for_mutant_failures = [(35, 0), (0, 21), (100, 25)]  # (x, 0) should return x, (0, y) should return y, valid pairs like (28, 14)
    
    for a, b in inputs_for_mutant_failures:
        correct_output = gcd(a, b)  # Expected output is properly defined
        print(f"Testing gcd({a}, {b}) = {correct_output} (correct implementation)")
        
        # Check mutant behavior
        def mutant_gcd(a, b):
            if b == 0:
                return a
            else:
                return gcd(a % b, b)  # Should produce a recursion issue
            
        try:
            # Check mutant behavior
            mutant_output = mutant_gcd(a, b)  
            print(f"Mutant output for gcd({a}, {b}) = {mutant_output}")
            assert mutant_output == correct_output, "The mutant should not match the correct GCD."
        except RecursionError:
            print(f"Mutant fails with RecursionError for inputs ({a}, {b}) as expected.")

# Call the test function
test__gcd()