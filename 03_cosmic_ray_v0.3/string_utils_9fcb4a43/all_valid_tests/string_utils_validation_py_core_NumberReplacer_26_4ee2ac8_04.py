from string_utils.validation import is_email

def test__is_email():
    """Focus on catching the mutant through clever use of RFC email structures."""
    
    # Valid email with unique character usage
    assert is_email("email+filter@example.com"), "Should be valid."
    
    # Valid email with unusual coloring
    assert is_email("user%domain@example.com"), "Email with % should be valid."
    
    # Invalid email structure with multiple successive dots
    assert not is_email("this..is@invalid.com"), "Should be invalid due to multiple consecutive dots."
    
    # Invalid email multiple '@' signs
    assert not is_email("invalid@@example.com"), "Should be invalid due to double @ signs."
    
    # Email with special characters placed incorrectly
    assert not is_email("username@.com"), "Should be invalid; cannot start with a dot."
    
    # Valid email submission at the limits of expected structures
    assert is_email("valid.email@example.com"), "Should be valid."
  
    # Edge case structure retaining 321 characters but in invalid form
    invalid_email_structure = "a" * 321 + "@example.com"  # 322 chars total
    assert not is_email(invalid_email_structure), "Emails that exceed structural validity must fail."

    # Confirming that a length-boundary email fails (e.g. just an extension longer in a way that messes with domain validation)
    invalid_boundary_email = "validemail" + "x" * 310 + "@example.com"  # exceeding limits of valid formation
    assert not is_email(invalid_boundary_email), "Should be invalid; mutant may incorrectly validate it."