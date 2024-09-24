from string_utils.generation import roman_range

def test__roman_range_final():
    """Validate that the implementation limits to valid ranges and raises errors appropriately."""
    try:
        # This should raise an error due to invalid step (0)
        roman_range(5, start=5, step=0)
        assert False, "Expected ValueError due to step being 0"
    except ValueError as e:
        assert str(e) == '"step" must be an integer in the range 1-3999', "Unexpected message for step=0"

    try:
        # This should also raise an error due to invalid negative step
        roman_range(5, start=4, step=4)
        assert False, "Expected OverflowError due to start/stop/step configuration"
    except OverflowError as e:
        assert str(e) == 'Invalid start/stop/step configuration', "Unexpected message for invalid configuration"

    try:
        # Checking proper behavior with equal start and stop
        output = list(roman_range(5, start=1, step=-1))  # Should raise an error as well
        assert False, "Expected OverflowError due to start/stop/step configuration"
    except OverflowError as e:
        assert str(e) == 'Invalid start/stop/step configuration', "Unexpected message for equal start/stop"

# Run the final test
test__roman_range_final()