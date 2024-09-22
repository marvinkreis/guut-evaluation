You are a helpful AI coding assistant, who is proficient in python coding and debugging.


We are going to give you a Python program and a mutant diff. We want you to write a test case that detects the mutant. The test case should pass when executed with the correct code, but fail when executed with the mutant.

- Output the test as single Python function called `test__<name>` with no parameters.
- Don't use any testing frameworks.
- Put your code in a markdown block and specify the language.
- Import all necessary files in your test code. You can assume that all python files we give you are in the root directory.
- Use assertions where appropriate.

Example:

```python
from sieve import sieve

def test_sieve()
    # test code here
```

Some mutants may be equivalent. Equivalent mutants don't change the behavior of the code, so they cannot be detected by a test. An example is changing `x=a+b` to `x=b+a`. If you believe a mutant to be equivalent, please claim the mutant as equivalent by writing the `# Equivalent Mutant` headline and giving a short explanation of why you think the mutant is equivalent.


# Task

```python string_utils/generation.py
001  # -*- coding: utf-8 -*-
002
003  # public api to export
004  __all__ = [
005      'uuid',
006      'random_string',
007      'secure_random_hex',
008      'roman_range',
009  ]
010
011  import binascii
012  import os
013  import random
014  import string
015  from typing import Generator
016  from uuid import uuid4
017
018  from .manipulation import roman_encode
019
020
021  def uuid(as_hex: bool = False) -> str:
022      """
023      Generated an UUID string (using `uuid.uuid4()`).
024
025      *Examples:*
026
027      >>> uuid() # possible output: '97e3a716-6b33-4ab9-9bb1-8128cb24d76b'
028      >>> uuid(as_hex=True) # possible output: '97e3a7166b334ab99bb18128cb24d76b'
029
030      :param as_hex: True to return the hex value of the UUID, False to get its default representation (default).
031      :return: uuid string.
032      """
033      uid = uuid4()
034
035      if as_hex:
036          return uid.hex
037
038      return str(uid)
039
040
041  def random_string(size: int) -> str:
042      """
043      Returns a string of the specified size containing random characters (uppercase/lowercase ascii letters and digits).
044
045      *Example:*
046
047      >>> random_string(9) # possible output: "cx3QQbzYg"
048
049      :param size: Desired string size
050      :type size: int
051      :return: Random string
052      """
053      if not isinstance(size, int) or size < 1:
054          raise ValueError('size must be >= 1')
055
056      chars = string.ascii_letters + string.digits
057      buffer = [random.choice(chars) for _ in range(size)]
058      out = ''.join(buffer)
059
060      return out
061
062
063  def secure_random_hex(byte_count: int) -> str:
064      """
065      Generates a random string using secure low level random generator (os.urandom).
066
067      **Bear in mind**: due to hex conversion, the returned string will have a size that is exactly\
068      the double of the given `byte_count`.
069
070      *Example:*
071
072      >>> secure_random_hex(9) # possible output: 'aac4cf1d1d87bd5036'
073
074      :param byte_count: Number of random bytes to generate
075      :type byte_count: int
076      :return: Hexadecimal string representation of generated random bytes
077      """
078      if not isinstance(byte_count, int) or byte_count < 1:
079          raise ValueError('byte_count must be >= 1')
080
081      random_bytes = os.urandom(byte_count)
082      hex_bytes = binascii.hexlify(random_bytes)
083      hex_string = hex_bytes.decode()
084
085      return hex_string
086
087
088  def roman_range(stop: int, start: int = 1, step: int = 1) -> Generator:
089      """
090      Similarly to native Python's `range()`, returns a Generator object which generates a new roman number
091      on each iteration instead of an integer.
092
093      *Example:*
094
095      >>> for n in roman_range(7): print(n)
096      >>> # prints: I, II, III, IV, V, VI, VII
097      >>> for n in roman_range(start=7, stop=1, step=-1): print(n)
098      >>> # prints: VII, VI, V, IV, III, II, I
099
100      :param stop: Number at which the generation must stop (must be <= 3999).
101      :param start: Number at which the generation must start (must be >= 1).
102      :param step: Increment of each generation step (default to 1).
103      :return: Generator of roman numbers.
104      """
105
106      def validate(arg_value, arg_name, allow_negative=False):
107          msg = '"{}" must be an integer in the range 1-3999'.format(arg_name)
108
109          if not isinstance(arg_value, int):
110              raise ValueError(msg)
111
112          if allow_negative:
113              arg_value = abs(arg_value)
114
115          if arg_value < 1 or arg_value > 3999:
116              raise ValueError(msg)
117
118      def generate():
119          current = start
120
121          # generate values for each step
122          while current != stop:
123              yield roman_encode(current)
124              current += step
125
126          # last value to return
127          yield roman_encode(current)
128
129      # checks each single argument value
130      validate(stop, 'stop')
131      validate(start, 'start')
132      validate(step, 'step', allow_negative=True)
133
134      # checks if the provided configuration leads to a feasible iteration with respect to boundaries or not
135      forward_exceed = step > 0 and (start > stop or start + step > stop)
136      backward_exceed = step < 0 and (start < stop or start + step < stop)
137      if forward_exceed or backward_exceed:
138          raise OverflowError('Invalid start/stop/step configuration')
139
140      return generate()
```

```diff mutant.diff
diff --git a/string_utils/generation.py b/mutant/string_utils/generation.py
index e470877..031a2c1 100644
--- a/string_utils/generation.py
+++ b/mutant/string_utils/generation.py
@@ -132,7 +132,7 @@ def roman_range(stop: int, start: int = 1, step: int = 1) -> Generator:
     validate(step, 'step', allow_negative=True)
 
     # checks if the provided configuration leads to a feasible iteration with respect to boundaries or not
-    forward_exceed = step > 0 and (start > stop or start + step > stop)
+    forward_exceed = step > 0 and (start > stop or start - step > stop)
     backward_exceed = step < 0 and (start < stop or start + step < stop)
     if forward_exceed or backward_exceed:
         raise OverflowError('Invalid start/stop/step configuration')
```

# Test


To detect the described mutant, we need to create a test case for the `roman_range` function. The mutant changes the condition for determining a "forward exceed", which will affect its behavior in specific scenarios, particularly when `step` is positive.

Here's a test case that will allow us to differentiate between the original code and the mutant:

```python
from generation import roman_range

def test_roman_range():
    # Test that checks the proper generation of roman numerals and handles step exceeding
    # Original case where step is greater leading to valid range
    romans = list(roman_range(5))
    assert romans == ['I', 'II', 'III', 'IV', 'V'], f"Expected ['I', 'II', 'III', 'IV', 'V'], got {romans}"

    # Test that checks overflow by starting at a position greater than stop with a positive step
    try:
        list(roman_range(1, start=5, step=1))
        assert False, "Expected OverflowError not raised for forward exceed, but it was."
    except OverflowError:
        pass  # Expected behavior, the error was raised

    # A valid scenario where step is negative
    rom_num = list(roman_range(stop=1, start=5, step=-1))
    assert rom_num == ['V', 'IV', 'III', 'II', 'I'], f"Expected ['V', 'IV', 'III', 'II', 'I'], got {rom_num}"

    # Another test for positive step but invalid configuration
    try:
        list(roman_range(stop=4, start=7, step=1))  # should raise an OverflowError
        assert False, "Expected OverflowError not raised for invalid forward exceed, but it was."
    except OverflowError:
        pass  # Expected behavior, the error was raised
```

### Explanation
1. The test checks if the function correctly generates a valid sequence of Roman numerals when the input is straightforward.
2. It checks if an `OverflowError` is raised in a scenario where the starting value is greater than the stopping value with a positive step (which should not yield valid output in the original logic).
3. It verifies that when moving backwards, the sequence is generated correctly.
4. A final check ensures that the conditions with a positive step which exceed the valid configuration correctly raise an `OverflowError`.

The mutant's conditional logic (`start - step > stop` instead of `start + step > stop`) changes which scenarios will correctly raise this error, allowing us to detect the mutant with these assertions.


### Test Results

### Output for Correct Code

```
Traceback (most recent call last):
File "test.py", line 1, in <module>
from generation import roman_range
ModuleNotFoundError: No module named 'generation'
```
The test exited with exit code 1.

### Output for Mutant

```
Traceback (most recent call last):
File "test.py", line 1, in <module>
from generation import roman_range
ModuleNotFoundError: No module named 'generation'
```
The test exited with exit code 1.

Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the correct code, and fail when executed with the mutant. Please adjust your test case.


It seems that the issue arises from the incorrect name used for importing the `roman_range` function from the `generation.py` module. The correct import format should reflect the actual file structure. Let's correct it and ensure that the test case properly distinguishes between the original and the mutant by maintaining the assertions and error handling appropriately.

Here's the updated test:

```python
from string_utils.generation import roman_range

def test_roman_range():
    # Test to check the proper generation of Roman numerals
    # Check normal functionality
    romans = list(roman_range(5))
    assert romans == ['I', 'II', 'III', 'IV', 'V'], f"Expected ['I', 'II', 'III', 'IV', 'V'], got {romans}"

    # This should raise an OverflowError when start > stop with a positive step
    try:
        list(roman_range(1, start=5, step=1))
        assert False, "Expected OverflowError not raised for invalid forward exceed, but it was."
    except OverflowError:
        pass  # Expected behavior, the error was raised

    # A valid case with a negative step
    romans_backward = list(roman_range(stop=1, start=5, step=-1))
    assert romans_backward == ['V', 'IV', 'III', 'II', 'I'], f"Expected ['V', 'IV', 'III', 'II', 'I'], got {romans_backward}"

    # Another test for positive step but invalid configuration
    try:
        list(roman_range(stop=4, start=7, step=1))  # should raise an OverflowError
        assert False, "Expected OverflowError not raised for invalid forward exceed, but it was."
    except OverflowError:
        pass  # Expected behavior, the error was raised
```

### Explanation of Changes Made
1. **Import Path:** The correct path for importing `roman_range` is specified as `from string_utils.generation import roman_range` to match the actual structure of the files.
2. **Assertion Logic:** The assertion logic remains the same to ensure we check the correct behavior of the function.

### Expected Behavior
- When executed with the correct code, the test should pass.
- When executed with the mutant (where the logic for checking forward exceed is altered), the test should fail specifically at the point where we expect an `OverflowError` that is now improperly configured.

This adjustment should help resolve the import error and correctly test for the mutant's logic change.


### Test Results

### Output for Correct Code

```

```

### Output for Mutant

```

```

Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the correct code, and fail when executed with the mutant. Please adjust your test case.


We need to ensure that the test case correctly identifies the change in behavior introduced by the mutant. The mutant alters the logic in the `forward_exceed` condition, and to effectively differentiate the correct code from the mutant, we can create a scenario that will only pass with the original code.

The strategy will be to use specific values for `start`, `stop`, and `step` that will allow us to create a situation where the logic differs between the original implementation and the mutant implementation.

Here's the revised test case:

```python
from string_utils.generation import roman_range

def test_roman_range():
    # Testing generation of Roman numerals correctly
    # This scenario should produce Roman numerals from I to V
    romans = list(roman_range(5))
    assert romans == ['I', 'II', 'III', 'IV', 'V'], f"Expected ['I', 'II', 'III', 'IV', 'V'], got {romans}"

    # This should raise an OverflowError because start is greater than stop with a step that cannot accommodate the range
    try:
        list(roman_range(10, start=5, step=1))  # This should trigger OverflowError for both original and mutant
        assert False, "Expected OverflowError not raised for invalid forward exceed, but it was."
    except OverflowError:
        pass  # Expected behavior

    # This scenario will help differentiate between correct logic and mutant logic
    # It checks that an attempt to generate starting from 1 to 3 with step 2 should produce an error
    try:
        list(roman_range(3, start=1, step=2))  # This should raise OverflowError because the step is too large
        assert False, "Expected OverflowError not raised for invalid range with step 2, but it was."
    except OverflowError:
        pass  # Expected behavior

    # Check a valid range with negative step
    romans_backward = list(roman_range(start=5, stop=1, step=-1))
    assert romans_backward == ['V', 'IV', 'III', 'II', 'I'], f"Expected ['V', 'IV', 'III', 'II', 'I'], got {romans_backward}"

    # This is to ensure we also correctly catch an invalid backward range
    try:
        list(roman_range(start=1, stop=4, step=-1))  # Should raise an OverflowError
        assert False, "Expected OverflowError not raised for invalid backward exceed, but it was."
    except OverflowError:
        pass  # Expected behavior
```

### How This Test Identifies the Mutant:
1. **Forward Exceed Scenarios:** The tests for `(stop=10, start=5, step=1)` should raise an `OverflowError` in both the original and mutant versions since it's clearly an invalid configuration.
2. **Logic Specificity:** The test case with `(stop=3, start=1, step=2)` is particularly crafted to ensure that the mutant, which has the altered logic, will handle this incorrectly. The original code checks if `start + step > stop`, which should trigger an overflow when `step` is greater than `stop - start`.
3. **Backward steps:** We check backward stepping scenarios to ensure that the validation applies consistently in both directions.

### Expected Outputs
- The original code should pass these tests with the expected assertions met.
- The mutant code will fail particularly on the test of the forward range that checks for a large step, as it will misinterpret the configuration due to the altered logic.

This setup should effectively differentiate the behavior of the original implementation from the mutant.


### Test Results

### Output for Correct Code

```
Traceback (most recent call last):
File "test.py", line 35, in <module>
test_roman_range()
File "test.py", line 12, in test_roman_range
assert False, "Expected OverflowError not raised for invalid forward exceed, but it was."
AssertionError: Expected OverflowError not raised for invalid forward exceed, but it was.
```
The test exited with exit code 1.

### Output for Mutant

```
Traceback (most recent call last):
File "test.py", line 35, in <module>
test_roman_range()
File "test.py", line 12, in test_roman_range
assert False, "Expected OverflowError not raised for invalid forward exceed, but it was."
AssertionError: Expected OverflowError not raised for invalid forward exceed, but it was.
```
The test exited with exit code 1.

Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the correct code, and fail when executed with the mutant. Please adjust your test case.


It looks like both the original code and the mutant failed to raise an OverflowError for the configurations provided, suggesting that there's a misunderstanding regarding what conditions induce that error. 

Let's clarify the conditions under which an `OverflowError` should be raised and create a test that effectively differentiates the behavior of the original implementation and the mutant based on those criteria.

To effectively identify the mutant, we need additional scenarios where the change in logic affects the outcome. In this case, let's focus on the conditions of generating an overflow more explicitly with better target conditions.

### Revised Test Case

Here’s a refined test case aimed at precisely capturing the behavior differences:

```python
from string_utils.generation import roman_range

def test_roman_range():
    # This should generate Romans from 1 to 5
    romans = list(roman_range(5))
    assert romans == ['I', 'II', 'III', 'IV', 'V'], f"Expected ['I', 'II', 'III', 'IV', 'V'], got {romans}"

    # Check a case that should pass normally
    try:
        output = list(roman_range(stop=10, start=1, step=2))  # This should produce valid output
        assert output == ['I', 'III', 'V', 'VII', 'IX'], f"Expected ['I', 'III', 'V', 'VII', 'IX'], got {output}"
    except OverflowError:
        assert False, "Expected not to raise OverflowError for valid range with step 2."

    # This should raise an OverflowError due to configuration (start=5, stop=1, step=1)
    try:
        list(roman_range(stop=1, start=5, step=1))  # Should raise OverflowError on original code
        assert False, "Expected OverflowError not raised for invalid forward exceed, but it was."
    except OverflowError:
        pass  # This is expected behavior for the original logic

    # Check a valid range going backward
    romans_backward = list(roman_range(start=5, stop=1, step=-1))
    assert romans_backward == ['V', 'IV', 'III', 'II', 'I'], f"Expected ['V', 'IV', 'III', 'II', 'I'], got {romans_backward}"

    # This should raise OverflowError for invalid backward range
    try:
        list(roman_range(stop=4, start=1, step=-1))  # Should raise OverflowError
        assert False, "Expected OverflowError not raised for invalid backward exceed, but it was."
    except OverflowError:
        pass  # This is expected

    # Adding a case specifically designed to identify mutant behavior
    # This case is for when using a step size that could cause the forward exceed condition
    try:
        list(roman_range(stop=10, start=1, step=20))  # This should raise OverflowError with the original code
        assert False, "Expected OverflowError not raised for very large positive step, but it was."
    except OverflowError:
        pass  # Expected due to the original logic
```

### Explanation of Changes:
1. **Direct Cases:** The first test checks for a legitimate generation of Roman numerals, ensuring we have valid execution for a straightforward case.
2. **Valid Output Check:** This is now added for `stop=10` and `start=1` with `step=2` to ensure valid processing under normal conditions.
3. **Specific Forward and Backward Checks:** There are specific tests for conditions that should definitely raise `OverflowError` in both the original code and mutant logic, focusing on edge cases.
4. **Mutant Detection:** The last check (`step=20`) should highlight the mutant's logic change, as it will behave differently from what the original implementation intends.

### Expected Results:
- The original code should pass the tests and correctly raise the exceptions where intended.
- The mutant should fail particularly on those checks that involve defining an invalid range with respects to the step configurations, illustrating the altered logic in handling the `forward_exceed` calculation.

With these focused adjustments, it should provide clearer detection of the mutant versus the original code.


### Test Results

### Output for Correct Code

```
Traceback (most recent call last):
File "test.py", line 41, in <module>
test_roman_range()
File "test.py", line 10, in test_roman_range
output = list(roman_range(stop=10, start=1, step=2))  # This should produce valid output
File "string_utils/generation.py", line 123, in generate
yield roman_encode(current)
File "string_utils/manipulation.py", line 634, in roman_encode
return __RomanNumbers.encode(input_number)
File "string_utils/manipulation.py", line 89, in encode
raise ValueError('Input must be >= 1 and <= 3999')
ValueError: Input must be >= 1 and <= 3999
```
The test exited with exit code 1.

### Output for Mutant

```
Traceback (most recent call last):
File "test.py", line 41, in <module>
test_roman_range()
File "test.py", line 10, in test_roman_range
output = list(roman_range(stop=10, start=1, step=2))  # This should produce valid output
File "string_utils/generation.py", line 123, in generate
yield roman_encode(current)
File "string_utils/manipulation.py", line 634, in roman_encode
return __RomanNumbers.encode(input_number)
File "string_utils/manipulation.py", line 89, in encode
raise ValueError('Input must be >= 1 and <= 3999')
ValueError: Input must be >= 1 and <= 3999
```
The test exited with exit code 1.

Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the correct code, and fail when executed with the mutant. Please adjust your test case.


The errors that are being raised indicate that the second test (the one checking `list(roman_range(stop=10, start=1, step=2))`) is improperly configured to fit within the expected range of inputs for Roman numeral encoding. This is indicating that there's a failure related to the parameters used when generating the Roman numerals.

Since the test case is not getting the expected responses from both the original and mutant installations, let’s adjust our approach. We need to ensure that our tests are constructed specifically to trigger the conditions relevant to the mutant, without causing unrelated errors due to the Roman numeral encoding process.

### Refined Testing Strategy

We will design the test cases more carefully, ensuring that we are testing scenarios that both the correct code and the mutant can adequately handle without running into exceptions that are unrelated to our condition checks.

Here’s the updated test case:

```python
from string_utils.generation import roman_range

def test_roman_range():
    # Test to check normal generation of Roman numerals.
    romans = list(roman_range(5))  # Generating Roman numerals from 1 to 5
    assert romans == ['I', 'II', 'III', 'IV', 'V'], f"Expected ['I', 'II', 'III', 'IV', 'V'], got {romans}"

    # Test with valid configuration
    output = list(roman_range(stop=7, start=1, step=1))  # Should generate I to VII
    assert output == ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'], f"Expected ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'], got {output}"

    # This should raise an OverflowError due to invalid parameters (start > stop)
    try:
        list(roman_range(stop=1, start=5, step=1))  # Should raise OverflowError
        assert False, "Expected OverflowError not raised for invalid step configuration, but it was."
    except OverflowError:
        pass  # This is expected behavior

    # Check valid backward stepping
    romans_backward = list(roman_range(start=5, stop=1, step=-1))  
    assert romans_backward == ['V', 'IV', 'III', 'II', 'I'], f"Expected ['V', 'IV', 'III', 'II', 'I'], got {romans_backward}"

    # Invalid backward range that should raise an OverflowError
    try:
        list(roman_range(stop=4, start=1, step=-1))  # Should raise OverflowError
        assert False, "Expected OverflowError not raised for invalid backward exceed, but it was."
    except OverflowError:
        pass  # This is expected

    # Specific case to identify mutant behavior
    try:
        list(roman_range(stop=10, start=1, step=10))  # Large step that should cause overflow
        assert False, "Expected OverflowError not raised for large step with forward exceed, but it was."
    except OverflowError:
        pass  # Expected due to overflow
    
    # Edge case for maximum valid bounds on Roman numerals
    try:
        output = list(roman_range(stop=3999, start=1, step=1))  # Should work well
        assert output[-1] == 'MMMCMXCIX', f"Expected last output to be 'MMMCMXCIX', got {output[-1]}"
    except Exception as e:
        assert False, f"Unexpected error for valid range: {str(e)}"
```

### Key Adjustments:
1. **Valid Test Cases:** It focuses on ensuring that tests produce valid outputs and includes valid start/stop configurations that are not risky.
2. **Systematic Testing for Overflow Conditions:** 
   - Set clear boundaries for what should trigger invalid configurations.
   - Ensure specific checks for very large step values that could misconfigure the expectations on output.
3. **Edge Case Handling:** It checks the high-end limits of the Roman numeral system to see if it reaches reasonable boundaries with no errors being raised.

### Expected Outcome:
- The original code will successfully pass these tests.
- The mutant version may fail on tests specifically checking conditions that lead to valid configurations becoming invalid due to the logic changes, especially on large step configurations. 

These refinements should lead to the pertinent differences between the original and mutant code without causing unrelated errors regarding numeral encoding.


# Results

The LLM found a test case that detects the mutant.

## Test Case

```python
from string_utils.generation import roman_range

def test_roman_range():
    # Test to check normal generation of Roman numerals.
    romans = list(roman_range(5))  # Generating Roman numerals from 1 to 5
    assert romans == ['I', 'II', 'III', 'IV', 'V'], f"Expected ['I', 'II', 'III', 'IV', 'V'], got {romans}"

    # Test with valid configuration
    output = list(roman_range(stop=7, start=1, step=1))  # Should generate I to VII
    assert output == ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'], f"Expected ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'], got {output}"

    # This should raise an OverflowError due to invalid parameters (start > stop)
    try:
        list(roman_range(stop=1, start=5, step=1))  # Should raise OverflowError
        assert False, "Expected OverflowError not raised for invalid step configuration, but it was."
    except OverflowError:
        pass  # This is expected behavior

    # Check valid backward stepping
    romans_backward = list(roman_range(start=5, stop=1, step=-1))  
    assert romans_backward == ['V', 'IV', 'III', 'II', 'I'], f"Expected ['V', 'IV', 'III', 'II', 'I'], got {romans_backward}"

    # Invalid backward range that should raise an OverflowError
    try:
        list(roman_range(stop=4, start=1, step=-1))  # Should raise OverflowError
        assert False, "Expected OverflowError not raised for invalid backward exceed, but it was."
    except OverflowError:
        pass  # This is expected

    # Specific case to identify mutant behavior
    try:
        list(roman_range(stop=10, start=1, step=10))  # Large step that should cause overflow
        assert False, "Expected OverflowError not raised for large step with forward exceed, but it was."
    except OverflowError:
        pass  # Expected due to overflow
    
    # Edge case for maximum valid bounds on Roman numerals
    try:
        output = list(roman_range(stop=3999, start=1, step=1))  # Should work well
        assert output[-1] == 'MMMCMXCIX', f"Expected last output to be 'MMMCMXCIX', got {output[-1]}"
    except Exception as e:
        assert False, f"Unexpected error for valid range: {str(e)}"
```

## Output for Correct Code

```

```

## Output for Mutant

```
Traceback (most recent call last):
File "test.py", line 44, in <module>
test_roman_range()
File "test.py", line 32, in test_roman_range
list(roman_range(stop=10, start=1, step=10))  # Large step that should cause overflow
File "string_utils/generation.py", line 123, in generate
yield roman_encode(current)
File "string_utils/manipulation.py", line 634, in roman_encode
return __RomanNumbers.encode(input_number)
File "string_utils/manipulation.py", line 89, in encode
raise ValueError('Input must be >= 1 and <= 3999')
ValueError: Input must be >= 1 and <= 3999
```
The test exited with exit code 1.
