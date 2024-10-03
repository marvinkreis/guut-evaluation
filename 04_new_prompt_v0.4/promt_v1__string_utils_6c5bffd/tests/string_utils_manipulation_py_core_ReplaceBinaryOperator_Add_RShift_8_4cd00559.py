from string_utils.manipulation import __StringFormatter

def test____ensure_left_space_only():
    """
    Test for the __ensure_left_space_only method of the __StringFormatter class. 
    The test checks if the mutant modification (using '>>' instead of '+') leads to 
    a different outcome compared to the baseline implementation. The test should raise
    an exception in the mutant case.
    """
    class MockMatch:
        def __init__(self, string):
            self.group = lambda x: string

    formatter = __StringFormatter("   test input  ")
    mock_match = MockMatch("   test input  ")

    # This should work in the baseline but fail in the mutant
    output = formatter._StringFormatter__ensure_left_space_only(mock_match)
    
    assert output == ' test input'  # This assertion is expected to succeed in baseline