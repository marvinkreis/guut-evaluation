from to_base import to_base

def test__to_base():
    # Test case to check the correct order of concatenation
    assert to_base(15, 16) == 'F'  # Expected output for base 16
    assert to_base(16, 16) == '10'  # Exactly checks the transition from 'F' to '10'
    
    # Check for binary conversion
    assert to_base(8, 2) == '1000'  # 8 in binary should be '1000'
    
    # Check for larger conversion
    assert to_base(31, 16) == '1F'  # Expected output, before for base 16
    assert to_base(255, 16) == 'FF'  # Checking hexadecimal
    
    # Test lower boundary
    assert to_base(1, 2) == '1'  # Ensure conversion of 1 works