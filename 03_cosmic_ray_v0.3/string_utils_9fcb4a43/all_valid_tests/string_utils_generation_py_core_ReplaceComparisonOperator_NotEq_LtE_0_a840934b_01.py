from string_utils.generation import roman_range

def test__roman_range():
    """Changing 'while current != stop' to 'while current <= stop' in roman_range would result in incorrect additional output."""
    
    # The expected outputs for stop=5 should not include "VI"
    output = list(roman_range(5))
    assert output == ['I', 'II', 'III', 'IV', 'V'], f"Expected ['I', 'II', 'III', 'IV', 'V'], got {output}"
    
    # The expected outputs for stop=3 should not include "IV"
    output_smaller = list(roman_range(3))
    assert output_smaller == ['I', 'II', 'III'], f"Expected ['I', 'II', 'III'], got {output_smaller}"