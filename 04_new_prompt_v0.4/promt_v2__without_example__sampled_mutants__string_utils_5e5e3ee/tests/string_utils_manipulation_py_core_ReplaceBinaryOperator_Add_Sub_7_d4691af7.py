from string_utils.manipulation import prettify

def test__prettify_mutant_killing():
    """
    Test to confirm that the __ensure_right_space_only method in the mutant raises a TypeError.
    The input has a character followed by a space which is expected to be affected by the mutation.
    The test will pass if a TypeError is raised in the mutant but not in the baseline.
    """
    input_string = "Hello world !"
    
    try:
        output = prettify(input_string)  # This should succeed in the baseline
        print(f"Output on baseline: {output}")
        assert True  # Confirm that we successfully ran in the baseline
    except TypeError:
        # If we are here, it means we caught a TypeError in the mutant case
        print("TypeError caught as expected on mutant.")
        assert False, "The test should not reach here on the mutant since it raised TypeError."