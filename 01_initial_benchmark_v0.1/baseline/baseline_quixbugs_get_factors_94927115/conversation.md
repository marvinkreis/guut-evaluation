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

```python get_factors.py
01  """
02  Prime Factorization
03
04
05  Factors an int using naive trial division.
06
07  Input:
08      n: An int to factor
09
10  Output:
11      A list of the prime factors of n in sorted order with repetition
12
13  Precondition:
14      n >= 1
15
16  Examples:
17      >>> get_factors(1)
18      []
19      >>> get_factors(100)
20      [2, 2, 5, 5]
21      >>> get_factors(101)
22      [101]
23  """
24
25  def get_factors(n):
26      if n == 1:
27          return []
28
29      for i in range(2, int(n ** 0.5) + 1):
30          if n % i == 0:
31              return [i] + get_factors(n // i)
32
33      return [n]
```

```diff mutant.diff
diff --git a/get_factors.py b/get_factors.py
index 88591b2..5d13617 100644
--- a/get_factors.py
+++ b/get_factors.py
@@ -30,4 +30,4 @@ def get_factors(n):
         if n % i == 0:
             return [i] + get_factors(n // i)
 
-    return [n]
+    return []
```

# Test


To create a test case that detects the mutant in the provided `get_factors` function, we need to design a test that will succeed with the original implementation but will fail due to the changes introduced in the mutant. 

The mutant changes the return value for numbers that are prime (or not composite) to an empty list instead of returning the number itself. 

Based on this, here's the test case:

```python
from get_factors import get_factors

def test_get_factors():
    # Test with a composite number (should return its prime factors)
    assert get_factors(12) == [2, 2, 3]  # Factors of 12 are 2 * 2 * 3
    
    # Test with a prime number (should return the number itself)
    assert get_factors(13) == [13]  # Factors of 13 are just 13 itself
    
    # Test with a known composite number (should return [2, 2, 5, 5])
    assert get_factors(100) == [2, 2, 5, 5]  # Factors of 100 are 2 * 2 * 5 * 5
    
    # Repeat the test to ensure consistency
    assert get_factors(101) == [101]  # Factors of 101 (a prime number)
```

### Explanation of the test case:
- The first assertion checks the factorization of 12, which is not expected to fail regardless.
- The second assertion tests a prime number (13) which should return [13].
- The third and fourth assertions validate the expected outcomes for other composite and prime numbers. The mutant will return an empty list for the prime number, causing this test to fail, while the original will succeed.

The original function's behavior is correctly validated with these assertions while the mutant with the change to `return []` will not satisfy the tests that require returning the prime number itself.


# Results

The LLM found a test case that detects the mutant.

## Test Case

```python
from get_factors import get_factors

def test_get_factors():
    # Test with a composite number (should return its prime factors)
    assert get_factors(12) == [2, 2, 3]  # Factors of 12 are 2 * 2 * 3
    
    # Test with a prime number (should return the number itself)
    assert get_factors(13) == [13]  # Factors of 13 are just 13 itself
    
    # Test with a known composite number (should return [2, 2, 5, 5])
    assert get_factors(100) == [2, 2, 5, 5]  # Factors of 100 are 2 * 2 * 5 * 5
    
    # Repeat the test to ensure consistency
    assert get_factors(101) == [101]  # Factors of 101 (a prime number)
```

## Test on correct code

```

```

## Test on mutant

```
Traceback (most recent call last):
File "test.py", line 16, in <module>
test_get_factors()
File "test.py", line 5, in test_get_factors
assert get_factors(12) == [2, 2, 3]  # Factors of 12 are 2 * 2 * 3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```
The test exited with exit code 1.
