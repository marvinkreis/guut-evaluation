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

```python rpn_eval.py
01  """
02  Reverse Polish Notation
03
04  Four-function calculator with input given in Reverse Polish Notation (RPN).
05
06  Input:
07      A list of values and operators encoded as floats and strings
08
09  Precondition:
10      all(
11          isinstance(token, float) or token in ('+', '-', '*', '/') for token in tokens
12      )
13
14  Example:
15      >>> rpn_eval([3.0, 5.0, '+', 2.0, '/'])
16      4.0
17  """
18
19  def rpn_eval(tokens):
20      def op(symbol, a, b):
21          return {
22              '+': lambda a, b: a + b,
23              '-': lambda a, b: a - b,
24              '*': lambda a, b: a * b,
25              '/': lambda a, b: a / b
26          }[symbol](a, b)
27
28      stack = []
29
30      for token in tokens:
31          if isinstance(token, float):
32              stack.append(token)
33          else:
34              a = stack.pop()
35              b = stack.pop()
36              stack.append(
37                  op(token, b, a)
38              )
39
40      return stack.pop()
```

```diff mutant.diff
diff --git a/rpn_eval.py b/rpn_eval.py
index 0103916..c1f9797 100644
--- a/rpn_eval.py
+++ b/rpn_eval.py
@@ -34,7 +34,7 @@ def rpn_eval(tokens):
             a = stack.pop()
             b = stack.pop()
             stack.append(
-                op(token, b, a)
+                op(token, a, b)
             )
 
     return stack.pop()
```

# Test


To detect the mutant that changes the order of arguments in the `op` function call from `op(token, b, a)` to `op(token, a, b)`, we will create a test case that will yield a different result based on the order of the operands in a subtraction operation.

Here's the test code that would detect this mutant:

```python
from rpn_eval import rpn_eval

def test_rpn_eval_subtraction_order():
    # This test will check if the function correctly evaluates
    # 5.0 - 3.0, which should give us 2.0 in proper order.
    result = rpn_eval([5.0, 3.0, '-'])
    assert result == 2.0, f"Expected 2.0 but got {result}"

    # Now, let’s validate the behavior of the mutant.
    # The mutant will evaluate it as 3.0 - 5.0, which should give us -2.0.
    # Since we cannot execute the mutant directly here, knowing its behavior is enough.
    # The assertion above will pass only if the original code works correctly.
```

When this test is run with the original `rpn_eval` code, it should pass with a result of `2.0`. If the mutant code is used instead, it will evaluate `3.0 - 5.0` and return `-2.0`, thereby failing the assertion.


# Results

The LLM found a test case that detects the mutant.

## Test Case

```python
from rpn_eval import rpn_eval

def test_rpn_eval_subtraction_order():
    # This test will check if the function correctly evaluates
    # 5.0 - 3.0, which should give us 2.0 in proper order.
    result = rpn_eval([5.0, 3.0, '-'])
    assert result == 2.0, f"Expected 2.0 but got {result}"

    # Now, let’s validate the behavior of the mutant.
    # The mutant will evaluate it as 3.0 - 5.0, which should give us -2.0.
    # Since we cannot execute the mutant directly here, knowing its behavior is enough.
    # The assertion above will pass only if the original code works correctly.
```

## Test on correct code

```

```

## Test on mutant

```
Traceback (most recent call last):
File "test.py", line 14, in <module>
test_rpn_eval_subtraction_order()
File "test.py", line 7, in test_rpn_eval_subtraction_order
assert result == 2.0, f"Expected 2.0 but got {result}"
^^^^^^^^^^^^^
AssertionError: Expected 2.0 but got -2.0
```
The test exited with exit code 1.
