from get_factors import get_factors

def test__get_factors_all_cases():
    """
    Test for a variety of inputs including prime (7, 13) and composite (30).
    The expected outputs will differ for the mutant and baseline, which will lead
    to assertion errors for the different cases tested below.
    """
    prime_cases = [7, 13]
    composite_cases = [30, 100]
    
    for n in prime_cases:
        output = get_factors(n)
        print(f"output for {n} = {output}")
        assert output == [n]

    for n in composite_cases:
        if n == 30:
            output = get_factors(n)
            print(f"output for {n} = {output}")
            assert output == [2, 3, 5]
        elif n == 100:
            output = get_factors(n)
            print(f"output for {n} = {output}")
            assert output == [2, 2, 5, 5]