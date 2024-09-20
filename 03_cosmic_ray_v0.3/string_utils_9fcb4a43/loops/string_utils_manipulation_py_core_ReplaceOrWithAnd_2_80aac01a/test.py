from string_utils.manipulation import compress

def test__compress_invalid_compression_level():
    """Test to ensure that invalid compression levels raise appropriate exceptions."""
    
    # Testing the correct implementation for compression_level=-1
    try:
        compress("valid string", compression_level=-1)
        assert False, "Correct implementation should raise ValueError for -1"
    except ValueError:
        pass  # This is expected

    # Testing the correct implementation for compression_level=10
    try:
        compress("valid string", compression_level=10)
        assert False, "Correct implementation should raise ValueError for 10"
    except ValueError:
        pass  # This is also expected

# Note: The line below is just a comment to denote where the mutant test would go
# In practice, you would run the mutant's implementation separately to see it pass this case