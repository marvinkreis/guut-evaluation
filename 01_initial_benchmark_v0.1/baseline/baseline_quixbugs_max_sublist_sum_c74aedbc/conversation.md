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

Some mutants may be equivalent. Equivalent mutants don't change the behavior of the code, so they cannot be detected by a test. An example is changing `x=a+b` to `x=b+a`. If you believe a mutant to be equivalent, please claim the mutant as equivalent by writing the `# Equivalent Mutant` headline and giving a short description of why you think the mutant is equivalent.


# Task

```python max_sublist_sum.py
01  """
02  Max Sublist Sum
03  max-sublist-sum
04
05  Efficient equivalent to max(sum(arr[i:j]) for 0 <= i <= j <= len(arr))
06
07  Algorithm source: WordAligned.org by Thomas Guest
08
09
10  Input:
11      arr: A list of ints
12
13  Output:
14      The maximum sublist sum
15
16  Example:
17      >>> max_sublist_sum([4, -5, 2, 1, -1, 3])
18      5
19  """
20
21  def max_sublist_sum(arr):
22      max_ending_here = 0
23      max_so_far = 0
24
25      for x in arr:
26          max_ending_here = max(0, max_ending_here + x)
27          max_so_far = max(max_so_far, max_ending_here)
28
29      return max_so_far
```

```diff mutant.diff
diff --git a/max_sublist_sum.py b/max_sublist_sum.py
index a1da87d..51eeaa7 100644
--- a/max_sublist_sum.py
+++ b/max_sublist_sum.py
@@ -23,7 +23,7 @@ def max_sublist_sum(arr):
     max_so_far = 0
 
     for x in arr:
-        max_ending_here = max(0, max_ending_here + x)
+        max_ending_here = max_ending_here + x
         max_so_far = max(max_so_far, max_ending_here)
 
     return max_so_far
```

# Test


Here's the test code to detect the mutant in the `max_sublist_sum.py` implementation. This test will pass with the original code, but will fail with the mutant due to the incorrect handling of negative sums.

```python
from max_sublist_sum import max_sublist_sum

def test_max_sublist_sum():
    # This is a specific case where handling of negative values should matter
    test_array = [4, -5, 2, 1, -1, 3]
    expected_result = 5  # The maximum sum of the subarray [2, 1, -1, 3]
    
    result = max_sublist_sum(test_array)
    
    # Assert that the function returns the expected maximum sublist sum
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
```

This test case specifically checks a condition where the functionality of correctly handling the maximum sum of subarrays, which includes negative values, is crucial. The mutant's change to line 26 disrupts this logic by accumulating values without resetting or properly managing the sum when it becomes negative, therefore failing the assertion in this scenario.


# Results

The LLM found a test case that detects the mutant.

## Test Case

```python
from max_sublist_sum import max_sublist_sum

def test_max_sublist_sum():
    # This is a specific case where handling of negative values should matter
    test_array = [4, -5, 2, 1, -1, 3]
    expected_result = 5  # The maximum sum of the subarray [2, 1, -1, 3]
    
    result = max_sublist_sum(test_array)
    
    # Assert that the function returns the expected maximum sublist sum
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
```

## Test on correct code

```

```

## Test on mutant

```
Traceback (most recent call last):
File "test.py", line 13, in <module>
test_max_sublist_sum()
File "test.py", line 11, in test_max_sublist_sum
assert result == expected_result, f"Expected {expected_result}, but got {result}"
^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 5, but got 4
```
The test exited with exit code 1.
