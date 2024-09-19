from bitcount import bitcount

def test__bitcount():
    """
    The change from '&' to '^' in bitcount will cause incorrect results
    for inputs with multiple 1-bits such as 3 or 15, which are expected to 
    terminate with a specific count of 1-bits.
    """
    # Input that will lead to different behavior in the mutant
    assert bitcount(3) == 2, "bitcount(3) must return 2"
    assert bitcount(15) == 4, "bitcount(15) must return 4"
    
    # Check for input that leads to infinite loop
    import time
    start_time = time.time()
    bitcount(1)  # Should terminate but results in infinite loop in mutant
    elapsed_time = time.time() - start_time
    
    assert elapsed_time < 1, "The mutant should not enter an infinite loop"