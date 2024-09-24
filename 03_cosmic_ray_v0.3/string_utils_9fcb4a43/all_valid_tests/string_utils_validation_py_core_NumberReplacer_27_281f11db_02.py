from string_utils.validation import is_email

def test__is_email():
    """Testing email validation with valid and invalid cases to identify mutant behavior."""
    
    # Edge email formats
    valid_email_edge = "a" * 63 + "@example.com"  # Should be valid

    # Invalid due to character limit (using +)
    invalid_email_long = "a" * 64 + "+test@example.com"  # Should fail in the mutant with change in limit
    
    # Valid maximum length handling
    valid_email_max = "simple+" + "string" * 20 + "@example.com"  # Clear valid

    # Test with assertions
    assert is_email(valid_email_edge), "Edge case should validate successfully."
    assert not is_email(invalid_email_long), "Mutant should fail due to character handling comparing to the correct."

    # Add another known good email for thorough coverage
    valid_email_simple = "valid.email@example.com"
    
    assert is_email(valid_email_simple), "Basic valid email should pass validation."

# Expected: 
# The correct code should pass the edge cases while the mutant will fail on the invalid case.