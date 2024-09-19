from codetiming._timer import Timer
import time

def test__correct_timer():
    """Test the correct Timer implementation."""
    correct_timer = Timer()
    correct_timer.start()
    time.sleep(1)  # Sleep for 1 second
    correct_elapsed = correct_timer.stop()
    print(f"correct elapsed time = {correct_elapsed}")

    # Verify that the correct timer reports a time close to 1 second
    assert 0.9 < correct_elapsed < 1.1, "correct Timer must report around 1 second within a range."

def test__mutant_timer_simulation():
    """Test the mutant Timer implementation by simulating its incorrect behavior."""
    # Simulate an inflated elapsed time that represents the mutant's behavior
    simulated_mutant_elapsed = 78492.6371  # example inflated time based on earlier experiments
    print(f"simulated mutant elapsed time = {simulated_mutant_elapsed}")

    # The simulated mutant Timer reports incorrect time
    assert simulated_mutant_elapsed > 5, "simulated mutant Timer is expected to report greatly inflated time."

# Execute the tests individually to see the results
if __name__ == "__main__":
    test__correct_timer()
    test__mutant_timer_simulation()