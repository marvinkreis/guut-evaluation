from wrap import wrap

def test_wrap_correct():
    """Tests the correct implementation of wrap."""
    text = "Hello, World! This is a test."
    cols = 10
    
    correct_output = wrap(text, cols)
    print(f"Correct Output: {correct_output}")
    
    # The output should maintain all parts of the string
    assert len(correct_output) == 4, "Correct implementation should yield 4 lines"

def test_wrap_mutant():
    """Simulates and tests the mutant implementation of wrap."""
    text = "Hello, World! This is a test."
    cols = 10
    
    # Mimicking the mutant's behavior by omitting the last line added
    mutant_output = wrap(text, cols)[:-1]
    print(f"Mutant Output: {mutant_output}")
    
    # The output will not include the last line which is " test."
    assert len(mutant_output) == 3, "Mutant implementation should yield 3 lines"