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

```python gcd.py
01  """
02  Input:
03      a: A nonnegative int
04      b: A nonnegative int
05
06
07  Greatest Common Divisor
08
09  Precondition:
10      isinstance(a, int) and isinstance(b, int)
11
12  Output:
13      The greatest int that divides evenly into a and b
14
15  Example:
16      >>> gcd(35, 21)
17      7
18
19  """
20
21  def gcd(a, b):
22      if b == 0:
23          return a
24      else:
25          return gcd(b, a % b)
```

```diff mutant.diff
diff --git a/gcd.py b/gcd.py
index b3f4a54..85b243e 100644
--- a/gcd.py
+++ b/gcd.py
@@ -22,4 +22,4 @@ def gcd(a, b):
     if b == 0:
         return a
     else:
-        return gcd(b, a % b)
+        return gcd(a % b, b)
```

# Test


```python
from gcd import gcd

def test_gcd():
    # The correct expected output for gcd(35, 21) is 7
    result = gcd(35, 21)
    assert result == 7, f"Expected 7, but got {result}"

    # This test case will help detect the mutant,
    # as the mutant gcd function will calculate gcd(21, 35)
    # which gives us an incorrect order of operations in the recursive calls.
    result_mutant = gcd(21, 35)
    # The correct gcd(21, 35) still results in 7, however,
    # it can be used to ensure the logic around order is functioning.
    assert result_mutant == 7, f"Expected 7, but got {result_mutant}"
```


# Results

The LLM found a test case that detects the mutant.

## Test Case

```python
from gcd import gcd

def test_gcd():
    # The correct expected output for gcd(35, 21) is 7
    result = gcd(35, 21)
    assert result == 7, f"Expected 7, but got {result}"

    # This test case will help detect the mutant,
    # as the mutant gcd function will calculate gcd(21, 35)
    # which gives us an incorrect order of operations in the recursive calls.
    result_mutant = gcd(21, 35)
    # The correct gcd(21, 35) still results in 7, however,
    # it can be used to ensure the logic around order is functioning.
    assert result_mutant == 7, f"Expected 7, but got {result_mutant}"
```

## Test on correct code

```

```

## Test on mutant

```
Traceback (most recent call last):
File "test.py", line 16, in <module>
test_gcd()
File "test.py", line 5, in test_gcd
result = gcd(35, 21)
^^^^^^^^^^^
File "gcd.py", line 25, in gcd
return gcd(a % b, b)
^^^^^^^^^^^^^
File "gcd.py", line 25, in gcd
return gcd(a % b, b)
^^^^^^^^^^^^^
File "gcd.py", line 25, in gcd
return gcd(a % b, b)
^^^^^^^^^^^^^
[Previous line repeated 995 more times]
RecursionError: maximum recursion depth exceeded
```
The test exited with exit code 1.
