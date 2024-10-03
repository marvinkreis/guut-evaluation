You are a helpful AI coding assistant, who is proficient in python coding and debugging.


We are going to give you a Python program and a mutant diff. We want you to use scientific debugging to gain an understanding of the mutant, and then write a test case that kills the mutant.

This is an automated process, consisting of a loop of "hypothesis", "experiment" and "conclusion" until you are ready to write a "test" or to declare the mutant "equivalent". During this loop, you will submit "experiment" code and "test" code, which our system is going to parse and then execute for you. Since your messages will be automatically parsed, pay close attention to the format we expect of your messages. This includes the markdown headlines (e.g., "# Experiment"). Do not write any markdown headlines other than the ones described below.


# Output Format

The process will use the following format:

    # Task
    (we will provide the code under test and the mutant)

    # Debugging

    ## Hypothesis
    (your hypothesis)

    ## Experiment
    (your experiment code and prediction)

    ### Experiment Results
    #### Running Experiment on Baseline
    (we will write the baseline results)
    #### Running Experiment on Mutant
    (we will write the mutant results)

    ## Conclusion
    (your conclusion)

    [repeat ("Hypothesis", "Experiment", "Experiment Results", "Conclusion") until you found inputs that can detect the mutant]

    ## Test
    (your mutant-killing test)

    ### Test Results
    #### Running Test on Baseline
    (we will give the results)
    #### Running Test on Mutant
    (we will give the results)

    [repeat ("Test") or ("Hypothesis", "Experiment", "Experiment Results", "Conclusion") until a test successfully kills the mutant]

    [at any point, if you believe the mutant to be equivalent to the original code]
    ## Equivalent Mutant
    (a short explanation about why the mutant is equivalent)

## Notes

Make sure that `## Experiment` is always followed by `### Experiment Results` and `## Test` is always followed by `## Test Results`. This is important for parsing your responses.


# Output Format for Code

Write all code in markdown code blocks and specify the language, e.g.,

    ```python
    // python code here
    ```

Make sure to import all necessary functions in every code snippet. You can assume that all python files we list are in the current directory (`.`). For example, you can import the following file with `import guut.config as config` or `from guut.config import example`:

```python guut/config.py
def example():
    pass
```

Output all code in single Python function called `test__<function_name>` with no parameters. Don't use any testing frameworks. Don't call the test function yourself.


# Running code

Whenever you submit a test case (experiment or test), our system will run your code on the **Baseline** (the correct code, without the mutant) and give you the output. When that is finished, it applies the **Mutant** to the code and runs your code again.

This means that your test case can only use one version of the target code (**Baseline** or **Mutant**) at once. Your test case simply imports the target code and runs it, without knowing if it imported the **Baseline** or the **Mutant**. Since the system will run your test case once with the **Baseline** and once with the **Mutant**, you will still get the output for both versions.

Again, you cannot import the **Baseline** and the **Mutant** together. Your tests will import the target code, which can be either the **Baseline** or the **Mutant**. It is your job to design the test in a way, such that it produces different outputs when we switch out the imported target code from the **Baseline** to the **Mutant**.

Therefore, there is no point in re-implementing the code yourself, since the test should examine the imported code, not a recreation of it. Recreating the target code will make your test case useless.

We also do not provide a `mutant` module, so imports like `from mutant.sieve import sieve` will result in `ModuleNotFoundError: No module named 'mutant'`. This makes the test case useless.

Our system also cannot respond to instructive comments like `# Please run this with the mutant:`. The system executes *your entire experiment/test* with the **Baseline**, then executes *your entire experiment/test* with the **Mutant**. Leaving instructions in comments won't change this, and will instead make your test case useless.

# Scientific Debugging

Scientific debugging is a systematic debugging approach based on the scientific method. The process follows a loop of:

- Hypothesis
- Experiment
- Conclusion

## Hypotheses

Each hypothesis should describe an assumption you have about the code. Hypotheses are the key aspect of scientific debugging, and should be written detailed and with great care.

- Base hypotheses on the findings of previous experiments.
- Don't repeat hypotheses you have already made.
- Don't base hypotheses on untested assumptions.

Predict exactly what will happen. Avoid broad predictions like "Under any of the given inputs, the mutant will behave differently". Instead, write predictions like "The input [input] will satisfy [conditions]

Hypotheses loosely follow this template: Given [observations], I hypothesize that [assumption] holds when [given inputs]. I predict that [assumed result] and I will verify this by [more explanation and experiment description].

## Experiments

After stating a hypothesis, you create an experiment to test it. Each experiment will contain a Python test case, which imports and calls the target code. Once you stated the test case, our system will add it to the target code and execute it. First, it runs your code on the **Baseline** (the correct code, without the mutant) and gives you the output. When that is finished, it applies the **Mutant** to the code and runs your code again, also giving you the output.

Each experiment should contain a relevant prediction based on your hypothesis and a way to verify that prediction based on the output. Basically, you run the target code and predict the output based on your hypothesis. Therefore, add print statements to print out relevant values, which will help you understand what the code is doing.

Your experiment is agnostic of which version of the code it is handling (**Baseline** or **Mutant**). Therefore, never use add print statements like `print(f"baseline output: {output}")` or `print(f"mutant output: {output}")`. This will make your experiment results confusing and useless. Instead, use print statements that make sense with both versions like `print(f"output: {output}")`.

Some notes:
- Keep your experiments/tests short and simple.
- Use print statements liberally in your experiments.
- Never recreate the mutant as part of your experiment/test.
- Check one input at a time.

Here is an example experiment for a `is_valid_parenthesization` function. See how it prints the output, then checks if the output matches the prediction.

### Example Experiment

```python
from is_valid_parenthesization import is_valid_parenthesization

def test__is_valid_parenthesization():
    """
    Check if the target function correctly handles missing closing parentheses.
    """
    output = is_valid_parenthesization('(()')
    assert output == False
```

## Running Experiment on Baseline

```

```

## Running Experiment on Mutant

```
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    test__is_valid_parenthesization()
  File "test.py", line 8, in test__is_valid_parenthesization
    assert output == False
           ^^^^^^^^^^^^^^^
AssertionError
```
The experiment exited with exit code 1.

## Conclusions

After every experiment, write a conclusion that summarizes on the results. Summarize your conclusion in a short list, so you can refer back to them easily.

Pay close attention to experiment output:
- Did the baseline have any errors? Does the experiment need to be fixed?
- Are there any discrepancies between the output of the **Baseline** and the **Mutant**? That means you detected mutant.

It is already enough to find a single input that can distinguish between the **Baseline** and the **Mutant**. Any difference in output counts, as well as any difference in exceptions or timeouts. Any difference in behavior. Once you have found an input that triggers a difference, you can continue and write the test case that fails when it observes the **Mutant** behavior.

Otherwise, keep creating hypotheses and experiments until you have found the right inputs. Then you can finish debugging and write the mutant-killing test.

## Tests

Once you have found any inputs that cause a difference in behavior, you can write a test that kills the mutant. Similarly to experiments, when you finished writing your code, we will copy the test case and execute it against the **Baseline**, i.e., the regular program without the mutant, then apply the **Mutant** and execute it again.

The test kills the mutant if, and only if, the test passes when executed with the **Baseline** and fails when executed with the **Mutant**. Here, failing is defined as exiting with exit code 1. This means that the test needs to result in either a *failed assertion*, an *uncaught exception* or a *timeout* when executed on the **Mutant**.

This means that you have to include relevant assertions in your test, unless the mutant raises an exception or results in a timeout. If the mutant raises an exception or error, do not use a `try-except` block to catch it. Adding a `try-except` block that handles the exception means that the exception will not make the test case fail.

Include a relevant docstring comment with a summary of your findings. The comment should explain what the test checks for and why. Include relevant findings from your conclusions.

Here is an example test for a `rpn_eval` function that evaluates expressions in Reverse Polish notation:

### Example Test

```python
from rpn_eval import rpn_eval

def test__rpn_eval():
    """
    Test whether operator arguments are interpreted in the correct order. The input represents the calculation (8 / 2),
    which will lead to different results if the argument order is swapped, since (2 / 8) != (8 / 2).
    """
    output = rpn_eval([8.0, 2.0, '/'])
    assert output == 4.0
```

#### Example Test Results

##### Running Test on Baseline
```

```

##### Running Test on Mutant
```
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    test__rpn_eval()
  File "test.py", line 8, in test__rpn_eval
    assert output == 4.0
           ^^^^^^^^^^^^^
AssertionError
```
The test exited with exit code 1.

## Equivalent Mutants

Some mutants may be equivalent. Equivalent mutants don't change the behavior of the code, meaning they cannot be detected by a test. An example would be changing `x=a+b` to `x=b+a`. If you believe a mutant to be equivalent, write the `## Equivalent Mutant` headline and give a short description of why you think the mutant is equivalent. Include some information from your experiments to back up your claims. Afterwards, try to prove yourself wrong by doing more experiments. See if you can maybe find a difference between the **Baseline** and the **Mutant** anyways.

Example:

I believe the mutant is equivalent. The change [mutant change] doesn't affect the way the target code computes [some result]. My previous tests show that [tried inputs] did not result in any different behavior in the mutant, which suggest [more explanation]. I will now try to detect the mutant anyways. Since my past tries have failed to detect the mutant, I will now try [new approach].

## Strategy

The first goal is always to cover the mutant, meaning to execute the part of the code it changes. Take a look at the target code and check which conditions your experiment/test needs to meet in order to execute the mutant (for example, passing parameters that meet certain if conditions), then create an experiment that meets those conditions.

Keep your experiments and tests short, so they don't fail because of accidental mistakes in your code. Try only one or two inputs at a time. Shorter experiments are also easier to understand and reason about.


# Task (Example)

```python find_in_sorted.py
01  """
02  Binary Search
03
04  Input:
05      arr: A sorted list of ints
06      x: A value to find
07
08  Output:
09      An index i such that arr[i] == x, or -1 if x not in arr
10
11  Example:
12      >>> find_in_sorted([3, 4, 5, 5, 5, 5, 6], 5)
13      3
14  """
15
16  def find_in_sorted(arr, x):
17      def binsearch(start, end):
18          if start == end:
19              return -1
20          mid = start + (end - start) // 2
21          if x < arr[mid]:
22              return binsearch(start, mid)
23          elif x > arr[mid]:
24              return binsearch(mid + 1, end)
25          else:
26              return mid
27
28      return binsearch(0, len(arr))
```

## Hypothesis (Example)
- The mutant changes a recursive call of the `binsearch` function from `binsearch(mid + 1, end)` to `binsearch(mid, end)`.
- The mutant is guarded by an if-condition that checks whether the search value is greater than `mid`.

I hypothesize that the mutant alters how the search space is reduced, making the function fail to find values that are present in the array. Specifically, I predict that the mutant will fail to find the search value if it is the highest value in the list, as the search value needs to be greater than `mid` in order to cover the mutant.

## Experiment (Example)
I will call `find_in_sorted` with `[1, 2, 3, 4]` as the list and `4` as the search value. I predict that the baseline will find the searched value in the list, while the mutant will fail to find the element and return `-1`.

```python
from find_in_sorted import find_in_sorted

def test_find_in_sorted():
    # Try finding the highest number at the end of the list.
    output = find_in_sorted([1, 2, 3, 4], 4)
    print(f"Output: {output}")
    assert output != -1
```

### Experiment Results (Example)

### Running Experiment on Baseline
```
Output: 3
```

### Running Experiment on Mutant
```
Output: 3
```

## Conclusion (Example)
The experiment shows that both the baseline and the mutant correctly find the `4` at the end of the list, returning the correct index `3`. This contradicts my prediction.

Key conclusions:
- Both baseline and mutant are able to find `4` in `[1, 2, 3, 4]`.

## Hypothesis (Example)
In the previous experiment, I tested whether an existing element can be found in the list. Now, I will check if searching an element that is not contained in the input list will result in a difference. I hypothesize that the mutant will give an incorrect output when a value greater than the highest value in the list is searched.

## Experiment (Example)
I will call `find_in_sorted` with `[1, 2, 3, 4]` as the list and `5` as the search value. I predict that the baseline will return `-1` as the value is not part of the list, while the mutant will return a different output.

```python
from find_in_sorted import find_in_sorted

def test_find_in_sorted():
    # Try finding a number higher than the highest number in the list.
    output = find_in_sorted([1, 2, 3, 4], 5)
    print(f"Output: {output}")
    assert output == -1, f"Expected -1, got {output}"
```

### Experiment Results (Example)

### Running Experiment on Baseline
```
Output: -1
```

### Running Experiment on Mutant
```
Traceback (most recent call last):
  File "test.py", line 8, in <module>
    test_find_in_sorted()
  File "test.py", line 5, in test_find_in_sorted
    output = find_in_sorted([1, 2, 3, 4], 5)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "find_in_sorted.py", line 28, in find_in_sorted
    return binsearch(0, len(arr))
           ^^^^^^^^^^^^^^^^^^^^^^
  File "find_in_sorted.py", line 24, in binsearch
    return binsearch(mid, end)
           ^^^^^^^^^^^^^^^^^^^
  [Previous line repeated 988 more times]
  File "find_in_sorted.py", line 17, in binsearch
    def binsearch(start, end):

RecursionError: maximum recursion depth exceeded
```
The experiment exited with exit code 1.

## Conclusion (Example)
The baseline returned `-1` as predicted, while the mutant results in a `RecursionError`. This confirms my hypothesis.

Key conclusions:
- The mutant raises an error when searching for `5` in `[1, 2, 3, 4]`.
- The baseline returns `-1` (the correct output) when searching for `5` in `[1, 2, 3, 4]`.

Since I have identified consistent behavior in the mutant that can be exploited, I am now ready to create a test that will kill the mutant.

## Test (Example)

The test will call `find_in_sorted` with `[1, 2, 3, 4]` as the list and `5` as the searched value. As demonstrated before, this should lead to a `RecursionError` for the mutant, while the baseline returns `-1`.

```python
from find_in_sorted import find_in_sorted

def test_find_in_sorted_mutant_killing():
    """
    Test the find_in_sorted function using a value greater than
    the highest in the array. The mutant will enter an infinite
    recursion, while the baseline will return -1.
    """
    output = find_in_sorted([1, 2, 3, 4], 5)
    assert output == -1, f"Expected -1, got {output}"
```

# Test Results (Example)

## Running Test on Baseline

```

```

## Running Test on Mutant

```
Traceback (most recent call last):
  File "test.py", line 11, in <module>
    test_find_in_sorted_mutant_killing()
  File "test.py", line 9, in test_find_in_sorted_mutant_killing
    output = find_in_sorted([1, 2, 3, 4], 5)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "find_in_sorted.py", line 28, in find_in_sorted
    return binsearch(0, len(arr))
           ^^^^^^^^^^^^^^^^^^^^^^
  File "find_in_sorted.py", line 24, in binsearch
    return binsearch(mid, end)
           ^^^^^^^^^^^^^^^^^^^
  File "find_in_sorted.py", line 24, in binsearch
    return binsearch(mid, end)
           ^^^^^^^^^^^^^^^^^^^
  File "find_in_sorted.py", line 24, in binsearch
    return binsearch(mid, end)
           ^^^^^^^^^^^^^^^^^^^
  [Previous line repeated 986 more times]
  File "find_in_sorted.py", line 17, in binsearch
    def binsearch(start, end):

RecursionError: maximum recursion depth exceeded
```
The test exited with exit code 1.


# Task

```python string_utils/manipulation.py
001  # -*- coding: utf-8 -*-
002
003  # public api to export
004  __all__ = [
005      'camel_case_to_snake',
006      'snake_case_to_camel',
007      'reverse',
008      'shuffle',
009      'strip_html',
010      'prettify',
011      'asciify',
012      'slugify',
013      'booleanize',
014      'strip_margin',
015      'compress',
016      'decompress',
017      'roman_encode',
018      'roman_decode',
019  ]
020
021  import base64
022  import random
023  import unicodedata
024  import zlib
025  from typing import Union
026  from uuid import uuid4
027
028  from ._regex import *
029  from .errors import InvalidInputError
030  from .validation import is_snake_case, is_full_string, is_camel_case, is_integer, is_string
031
032
033  # PRIVATE API
034
035
036  class __RomanNumbers:
037      # internal rule mappings for encode()
038      __mappings = [
039          # units
040          {1: 'I', 5: 'V'},
041          # tens
042          {1: 'X', 5: 'L'},
043          # hundreds
044          {1: 'C', 5: 'D'},
045          # thousands
046          {1: 'M'},
047      ]
048
049      # swap key/value definitions for decode()
050      __reversed_mappings = [{v: k for k, v in m.items()} for m in __mappings]
051
052      @classmethod
053      def __encode_digit(cls, index: int, value: int) -> str:
054          # if digit is zero, there is no sign to display
055          if value == 0:
056              return ''
057
058          # from 1 to 3 we have just to repeat the sign N times (eg: III, XXX...)
059          if value <= 3:
060              return cls.__mappings[index][1] * value
061
062          # if 4 we have to add unit prefix
063          if value == 4:
064              return cls.__mappings[index][1] + cls.__mappings[index][5]
065
066          # if is 5, is a straight map
067          if value == 5:
068              return cls.__mappings[index][5]
069
070          # if 6, 7 or 8 we have to append unit suffixes
071          if value <= 8:
072              suffix = cls.__mappings[index][1] * (value - 5)
073              return cls.__mappings[index][5] + suffix
074
075          # if 9 we have to prepend current unit to next
076          return cls.__mappings[index][1] + cls.__mappings[index + 1][1]
077
078      @classmethod
079      def encode(cls, input_number: Union[str, int]) -> str:
080          # force input conversion to a string (we need it in order to iterate on each digit)
081          input_string = str(input_number)
082
083          if not is_integer(input_string):
084              raise ValueError('Invalid input, only strings or integers are allowed')
085
086          value = int(input_string)
087
088          if value < 1 or value > 3999:
089              raise ValueError('Input must be >= 1 and <= 3999')
090
091          input_len = len(input_string)
092          output = ''
093
094          # decode digits from right to left (start from units to thousands)
095          for index in range(input_len):
096              # get actual digit value as int
097              digit = int(input_string[input_len - index - 1])
098
099              # encode digit to roman string
100              encoded_digit = cls.__encode_digit(index, digit)
101
102              # prepend encoded value to the current output in order to have the final string sorted
103              # from thousands to units
104              output = encoded_digit + output
105
106          return output
107
108      @classmethod
109      def __index_for_sign(cls, sign: str) -> int:
110          for index, mapping in enumerate(cls.__reversed_mappings):
111              if sign in mapping:
112                  return index
113
114          raise ValueError('Invalid token found: "{}"'.format(sign))
115
116      @classmethod
117      def decode(cls, input_string: str) -> int:
118          if not is_full_string(input_string):
119              raise ValueError('Input must be a non empty string')
120
121          # reverse the provided string so that we can start parsing from units to thousands
122          reversed_string = reverse(input_string.upper())
123
124          # track last used value
125          last_value = None
126
127          # computed number to return
128          output = 0
129
130          # for each sign in the string we get its numeric value and add or subtract it to the computed output
131          for sign in reversed_string:
132              # are we dealing with units, tens, hundreds or thousands?
133              index = cls.__index_for_sign(sign)
134
135              # it's basically 1 or 5 (based on mapping rules definitions)
136              key_value = cls.__reversed_mappings[index][sign]
137
138              # Based on the level (tens, hundreds...) we have to add as many zeroes as the level into which we are
139              # in order to have the actual sign value.
140              # For instance, if we are at level 2 we are dealing with hundreds, therefore instead of 1 or 5, we will
141              # obtain 100 or 500 by adding 2 zeroes
142              sign_value = int(str(key_value) + '0' * index)
143
144              # increase total value if we are moving on with level
145              if last_value is None or sign_value >= last_value:
146                  output += sign_value
147
148              # Decrease value if we are back to a previous level
149              # For instance, if we are parsing "IX", we first encounter "X" which is ten then "I" which is unit,
150              # So we have to do the following operation in order to get 9 (the final result): 10 - 1
151              else:
152                  output -= sign_value
153
154              last_value = sign_value
155
156          return output
157
158
159  class __StringCompressor:
160
161      @staticmethod
162      def __require_valid_input_and_encoding(input_string: str, encoding: str):
163          if not is_string(input_string):
164              raise InvalidInputError(input_string)
165
166          if len(input_string) == 0:
167              raise ValueError('Input string cannot be empty')
168
169          if not is_string(encoding):
170              raise ValueError('Invalid encoding')
171
172      @classmethod
173      def compress(cls, input_string: str, encoding: str = 'utf-8', compression_level: int = 9) -> str:
174          cls.__require_valid_input_and_encoding(input_string, encoding)
175
176          if not isinstance(compression_level, int) or compression_level < 0 or compression_level > 9:
177              raise ValueError('Invalid compression_level: it must be an "int" between 0 and 9')
178
179          # turns input string into a sequence of bytes using provided encoding
180          original_bytes = input_string.encode(encoding)
181
182          # compress bytes using zlib library
183          compressed_bytes = zlib.compress(original_bytes, compression_level)
184
185          # encode compressed bytes using base64
186          # (this ensure that all characters will be available and that the output string can be used safely in any
187          # context such URLs)
188          encoded_bytes = base64.urlsafe_b64encode(compressed_bytes)
189
190          # finally turns base64 bytes into a string
191          output = encoded_bytes.decode(encoding)
192
193          return output
194
195      @classmethod
196      def decompress(cls, input_string: str, encoding: str = 'utf-8') -> str:
197          cls.__require_valid_input_and_encoding(input_string, encoding)
198
199          # turns input string into a sequence of bytes
200          # (the string is assumed to be a previously compressed string, therefore we have to decode it using base64)
201          input_bytes = base64.urlsafe_b64decode(input_string)
202
203          # decompress bytes using zlib
204          decompressed_bytes = zlib.decompress(input_bytes)
205
206          # decode the decompressed bytes to get the original string back
207          original_string = decompressed_bytes.decode(encoding)
208
209          return original_string
210
211
212  class __StringFormatter:
213      def __init__(self, input_string):
214          if not is_string(input_string):
215              raise InvalidInputError(input_string)
216
217          self.input_string = input_string
218
219      def __uppercase_first_char(self, regex_match):
220          return regex_match.group(0).upper()
221
222      def __remove_duplicates(self, regex_match):
223          return regex_match.group(1)[0]
224
225      def __uppercase_first_letter_after_sign(self, regex_match):
226          match = regex_match.group(1)
227          return match[:-1] + match[2].upper()
228
229      def __ensure_right_space_only(self, regex_match):
230          return regex_match.group(1).strip() + ' '
231
232      def __ensure_left_space_only(self, regex_match):
233          return ' ' + regex_match.group(1).strip()
234
235      def __ensure_spaces_around(self, regex_match):
236          return ' ' + regex_match.group(1).strip() + ' '
237
238      def __remove_internal_spaces(self, regex_match):
239          return regex_match.group(1).strip()
240
241      def __fix_saxon_genitive(self, regex_match):
242          return regex_match.group(1).replace(' ', '') + ' '
243
244      # generates a placeholder to inject temporary into the string, it will be replaced with the original
245      # value at the end of the process
246      @staticmethod
247      def __placeholder_key():
248          return '$' + uuid4().hex + '$'
249
250      def format(self) -> str:
251          # map of temporary placeholders
252          placeholders = {}
253          out = self.input_string
254
255          # looks for url or email and updates placeholders map with found values
256          placeholders.update({self.__placeholder_key(): m[0] for m in URLS_RE.findall(out)})
257          placeholders.update({self.__placeholder_key(): m for m in EMAILS_RE.findall(out)})
258
259          # replace original value with the placeholder key
260          for p in placeholders:
261              out = out.replace(placeholders[p], p, 1)
262
263          out = PRETTIFY_RE['UPPERCASE_FIRST_LETTER'].sub(self.__uppercase_first_char, out)
264          out = PRETTIFY_RE['DUPLICATES'].sub(self.__remove_duplicates, out)
265          out = PRETTIFY_RE['RIGHT_SPACE'].sub(self.__ensure_right_space_only, out)
266          out = PRETTIFY_RE['LEFT_SPACE'].sub(self.__ensure_left_space_only, out)
267          out = PRETTIFY_RE['SPACES_AROUND'].sub(self.__ensure_spaces_around, out)
268          out = PRETTIFY_RE['SPACES_INSIDE'].sub(self.__remove_internal_spaces, out)
269          out = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].sub(self.__uppercase_first_letter_after_sign, out)
270          out = PRETTIFY_RE['SAXON_GENITIVE'].sub(self.__fix_saxon_genitive, out)
271          out = out.strip()
272
273          # restore placeholder keys with their associated original value
274          for p in placeholders:
275              out = out.replace(p, placeholders[p], 1)
276
277          return out
278
279
280  # PUBLIC API
281
282  def reverse(input_string: str) -> str:
283      """
284      Returns the string with its chars reversed.
285
286      *Example:*
287
288      >>> reverse('hello') # returns 'olleh'
289
290      :param input_string: String to revert.
291      :type input_string: str
292      :return: Reversed string.
293      """
294      if not is_string(input_string):
295          raise InvalidInputError(input_string)
296
297      return input_string[::-1]
298
299
300  def camel_case_to_snake(input_string, separator='_'):
301      """
302      Convert a camel case string into a snake case one.
303      (The original string is returned if is not a valid camel case string)
304
305      *Example:*
306
307      >>> camel_case_to_snake('ThisIsACamelStringTest') # returns 'this_is_a_camel_case_string_test'
308
309      :param input_string: String to convert.
310      :type input_string: str
311      :param separator: Sign to use as separator.
312      :type separator: str
313      :return: Converted string.
314      """
315      if not is_string(input_string):
316          raise InvalidInputError(input_string)
317
318      if not is_camel_case(input_string):
319          return input_string
320
321      return CAMEL_CASE_REPLACE_RE.sub(lambda m: m.group(1) + separator, input_string).lower()
322
323
324  def snake_case_to_camel(input_string: str, upper_case_first: bool = True, separator: str = '_') -> str:
325      """
326      Convert a snake case string into a camel case one.
327      (The original string is returned if is not a valid snake case string)
328
329      *Example:*
330
331      >>> snake_case_to_camel('the_snake_is_green') # returns 'TheSnakeIsGreen'
332
333      :param input_string: String to convert.
334      :type input_string: str
335      :param upper_case_first: True to turn the first letter into uppercase (default).
336      :type upper_case_first: bool
337      :param separator: Sign to use as separator (default to "_").
338      :type separator: str
339      :return: Converted string
340      """
341      if not is_string(input_string):
342          raise InvalidInputError(input_string)
343
344      if not is_snake_case(input_string, separator):
345          return input_string
346
347      tokens = [s.title() for s in input_string.split(separator) if is_full_string(s)]
348
349      if not upper_case_first:
350          tokens[0] = tokens[0].lower()
351
352      out = ''.join(tokens)
353
354      return out
355
356
357  def shuffle(input_string: str) -> str:
358      """
359      Return a new string containing same chars of the given one but in a randomized order.
360
361      *Example:*
362
363      >>> shuffle('hello world') # possible output: 'l wodheorll'
364
365      :param input_string: String to shuffle
366      :type input_string: str
367      :return: Shuffled string
368      """
369      if not is_string(input_string):
370          raise InvalidInputError(input_string)
371
372      # turn the string into a list of chars
373      chars = list(input_string)
374
375      # shuffle the list
376      random.shuffle(chars)
377
378      # convert the shuffled list back to string
379      return ''.join(chars)
380
381
382  def strip_html(input_string: str, keep_tag_content: bool = False) -> str:
383      """
384      Remove html code contained into the given string.
385
386      *Examples:*
387
388      >>> strip_html('test: <a href="foo/bar">click here</a>') # returns 'test: '
389      >>> strip_html('test: <a href="foo/bar">click here</a>', keep_tag_content=True) # returns 'test: click here'
390
391      :param input_string: String to manipulate.
392      :type input_string: str
393      :param keep_tag_content: True to preserve tag content, False to remove tag and its content too (default).
394      :type keep_tag_content: bool
395      :return: String with html removed.
396      """
397      if not is_string(input_string):
398          raise InvalidInputError(input_string)
399
400      r = HTML_TAG_ONLY_RE if keep_tag_content else HTML_RE
401
402      return r.sub('', input_string)
403
404
405  def prettify(input_string: str) -> str:
406      """
407      Reformat a string by applying the following basic grammar and formatting rules:
408
409      - String cannot start or end with spaces
410      - The first letter in the string and the ones after a dot, an exclamation or a question mark must be uppercase
411      - String cannot have multiple sequential spaces, empty lines or punctuation (except for "?", "!" and ".")
412      - Arithmetic operators (+, -, /, \\*, =) must have one, and only one space before and after themselves
413      - One, and only one space should follow a dot, a comma, an exclamation or a question mark
414      - Text inside double quotes cannot start or end with spaces, but one, and only one space must come first and \
415      after quotes (foo" bar"baz -> foo "bar" baz)
416      - Text inside round brackets cannot start or end with spaces, but one, and only one space must come first and \
417      after brackets ("foo(bar )baz" -> "foo (bar) baz")
418      - Percentage sign ("%") cannot be preceded by a space if there is a number before ("100 %" -> "100%")
419      - Saxon genitive is correct ("Dave' s dog" -> "Dave's dog")
420
421      *Examples:*
422
423      >>> prettify(' unprettified string ,, like this one,will be"prettified" .it\\' s awesome! ')
424      >>> # -> 'Unprettified string, like this one, will be "prettified". It\'s awesome!'
425
426      :param input_string: String to manipulate
427      :return: Prettified string.
428      """
429      formatted = __StringFormatter(input_string).format()
430      return formatted
431
432
433  def asciify(input_string: str) -> str:
434      """
435      Force string content to be ascii-only by translating all non-ascii chars into the closest possible representation
436      (eg: ó -> o, Ë -> E, ç -> c...).
437
438      **Bear in mind**: Some chars may be lost if impossible to translate.
439
440      *Example:*
441
442      >>> asciify('èéùúòóäåëýñÅÀÁÇÌÍÑÓË') # returns 'eeuuooaaeynAAACIINOE'
443
444      :param input_string: String to convert
445      :return: Ascii utf-8 string
446      """
447      if not is_string(input_string):
448          raise InvalidInputError(input_string)
449
450      # "NFKD" is the algorithm which is able to successfully translate the most of non-ascii chars
451      normalized = unicodedata.normalize('NFKD', input_string)
452
453      # encode string forcing ascii and ignore any errors (unrepresentable chars will be stripped out)
454      ascii_bytes = normalized.encode('ascii', 'ignore')
455
456      # turns encoded bytes into an utf-8 string
457      ascii_string = ascii_bytes.decode('utf-8')
458
459      return ascii_string
460
461
462  def slugify(input_string: str, separator: str = '-') -> str:
463      """
464      Converts a string into a "slug" using provided separator.
465      The returned string has the following properties:
466
467      - it has no spaces
468      - all letters are in lower case
469      - all punctuation signs and non alphanumeric chars are removed
470      - words are divided using provided separator
471      - all chars are encoded as ascii (by using `asciify()`)
472      - is safe for URL
473
474      *Examples:*
475
476      >>> slugify('Top 10 Reasons To Love Dogs!!!') # returns: 'top-10-reasons-to-love-dogs'
477      >>> slugify('Mönstér Mägnët') # returns 'monster-magnet'
478
479      :param input_string: String to convert.
480      :type input_string: str
481      :param separator: Sign used to join string tokens (default to "-").
482      :type separator: str
483      :return: Slug string
484      """
485      if not is_string(input_string):
486          raise InvalidInputError(input_string)
487
488      # replace any character that is NOT letter or number with spaces
489      out = NO_LETTERS_OR_NUMBERS_RE.sub(' ', input_string.lower()).strip()
490
491      # replace spaces with join sign
492      out = SPACES_RE.sub(separator, out)
493
494      # normalize joins (remove duplicates)
495      out = re.sub(re.escape(separator) + r'+', separator, out)
496
497      return asciify(out)
498
499
500  def booleanize(input_string: str) -> bool:
501      """
502      Turns a string into a boolean based on its content (CASE INSENSITIVE).
503
504      A positive boolean (True) is returned if the string value is one of the following:
505
506      - "true"
507      - "1"
508      - "yes"
509      - "y"
510
511      Otherwise False is returned.
512
513      *Examples:*
514
515      >>> booleanize('true') # returns True
516      >>> booleanize('YES') # returns True
517      >>> booleanize('nope') # returns False
518
519      :param input_string: String to convert
520      :type input_string: str
521      :return: True if the string contains a boolean-like positive value, false otherwise
522      """
523      if not is_string(input_string):
524          raise InvalidInputError(input_string)
525
526      return input_string.lower() in ('true', '1', 'yes', 'y')
527
528
529  def strip_margin(input_string: str) -> str:
530      """
531      Removes tab indentation from multi line strings (inspired by analogous Scala function).
532
533      *Example:*
534
535      >>> strip_margin('''
536      >>>                 line 1
537      >>>                 line 2
538      >>>                 line 3
539      >>> ''')
540      >>> # returns:
541      >>> '''
542      >>> line 1
543      >>> line 2
544      >>> line 3
545      >>> '''
546
547      :param input_string: String to format
548      :type input_string: str
549      :return: A string without left margins
550      """
551      if not is_string(input_string):
552          raise InvalidInputError(input_string)
553
554      line_separator = '\n'
555      lines = [MARGIN_RE.sub('', line) for line in input_string.split(line_separator)]
556      out = line_separator.join(lines)
557
558      return out
559
560
561  def compress(input_string: str, encoding: str = 'utf-8', compression_level: int = 9) -> str:
562      """
563      Compress the given string by returning a shorter one that can be safely used in any context (like URL) and
564      restored back to its original state using `decompress()`.
565
566      **Bear in mind:**
567      Besides the provided `compression_level`, the compression result (how much the string is actually compressed
568      by resulting into a shorter string) depends on 2 factors:
569
570      1. The amount of data (string size): short strings might not provide a significant compression result\
571      or even be longer than the given input string (this is due to the fact that some bytes have to be embedded\
572      into the compressed string in order to be able to restore it later on)\
573
574      2. The content type: random sequences of chars are very unlikely to be successfully compressed, while the best\
575      compression result is obtained when the string contains several recurring char sequences (like in the example).
576
577      Behind the scenes this method makes use of the standard Python's zlib and base64 libraries.
578
579      *Examples:*
580
581      >>> n = 0 # <- ignore this, it's a fix for Pycharm (not fixable using ignore comments)
582      >>> # "original" will be a string with 169 chars:
583      >>> original = ' '.join(['word n{}'.format(n) for n in range(20)])
584      >>> # "compressed" will be a string of 88 chars
585      >>> compressed = compress(original)
586
587      :param input_string: String to compress (must be not empty or a ValueError will be raised).
588      :type input_string: str
589      :param encoding: String encoding (default to "utf-8").
590      :type encoding: str
591      :param compression_level: A value between 0 (no compression) and 9 (best compression), default to 9.
592      :type compression_level: int
593      :return: Compressed string.
594      """
595      return __StringCompressor.compress(input_string, encoding, compression_level)
596
597
598  def decompress(input_string: str, encoding: str = 'utf-8') -> str:
599      """
600      Restore a previously compressed string (obtained using `compress()`) back to its original state.
601
602      :param input_string: String to restore.
603      :type input_string: str
604      :param encoding: Original string encoding.
605      :type encoding: str
606      :return: Decompressed string.
607      """
608      return __StringCompressor.decompress(input_string, encoding)
609
610
611  def roman_encode(input_number: Union[str, int]) -> str:
612      """
613      Convert the given number/string into a roman number.
614
615      The passed input must represents a positive integer in the range 1-3999 (inclusive).
616
617      Why this limit? You may be wondering:
618
619      1. zero is forbidden since there is no related representation in roman numbers
620      2. the upper bound 3999 is due to the limitation in the ascii charset\
621      (the higher quantity sign displayable in ascii is "M" which is equal to 1000, therefore based on\
622      roman numbers rules we can use 3 times M to reach 3000 but we can't go any further in thousands without\
623      special "boxed chars").
624
625      *Examples:*
626
627      >>> roman_encode(37) # returns 'XXXVIII'
628      >>> roman_encode('2020') # returns 'MMXX'
629
630      :param input_number: An integer or a string to be converted.
631      :type input_number: Union[str, int]
632      :return: Roman number string.
633      """
634      return __RomanNumbers.encode(input_number)
635
636
637  def roman_decode(input_string: str) -> int:
638      """
639      Decode a roman number string into an integer if the provided string is valid.
640
641      *Example:*
642
643      >>> roman_decode('VII') # returns 7
644
645      :param input_string: (Assumed) Roman number
646      :type input_string: str
647      :return: Integer value
648      """
649      return __RomanNumbers.decode(input_string)
```

```diff mutant.diff
diff --git a/string_utils/manipulation.py b/mutant/string_utils/manipulation.py
index 780de34..8f6e823 100644
--- a/string_utils/manipulation.py
+++ b/mutant/string_utils/manipulation.py
@@ -228,11 +228,11 @@ class __StringFormatter:
 
     def __ensure_right_space_only(self, regex_match):
         return regex_match.group(1).strip() + ' '
 
     def __ensure_left_space_only(self, regex_match):
-        return ' ' + regex_match.group(1).strip()
+        return ' ' | regex_match.group(1).strip()
 
     def __ensure_spaces_around(self, regex_match):
         return ' ' + regex_match.group(1).strip() + ' '
 
     def __remove_internal_spaces(self, regex_match):
```

# Debugging


## Hypothesis
The mutant modifies the `__ensure_left_space_only` method by changing the concatenation operator from `+` to `|`, which is a bitwise OR operator. Given this modification, I hypothesize that the mutant will not return a correctly formatted string when whitespace is intended. Specifically, I predict that for any input string that aims to ensure a left space only (i.e., starting the string with one space), the mutant will return a different result compared to the baseline, especially when the input string is non-empty and contains leading spaces.

## Experiment
I will test the `prettify` function with an input string that has leading whitespace to verify if the output differs between the baseline and the mutant. The input will be a string that has leading whitespace followed by additional characters. I predict that the baseline will return a properly formatted string starting with a single space, while the mutant will return an incorrect string due to the erroneous use of the bitwise OR operator.

```python
from string_utils.manipulation import prettify

def test_prettify():
    # Input string with leading space
    input_string = '   this is a test string.'  # leading spaces should be trimmed to one
    output = prettify(input_string)
    print(f"Output: '{output}'")  # Output should start with one space
    assert output == ' this is a test string.', f"Unexpected output: '{output}'"
```


### Experiment Results

### Running Experiment on Baseline
```
Output: 'This is a test string.'
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    test_prettify()
  File "test.py", line 8, in test_prettify
    assert output == ' this is a test string.', f"Unexpected output: '{output}'"
AssertionError: Unexpected output: 'This is a test string.'
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
Output: 'This is a test string.'
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    test_prettify()
  File "test.py", line 8, in test_prettify
    assert output == ' this is a test string.', f"Unexpected output: '{output}'"
AssertionError: Unexpected output: 'This is a test string.'
```
The experiment exited with exit code 1.


## Conclusion
The experiment indicates that both the baseline and the mutant return the output `'This is a test string.'` when provided with the input string `'   this is a test string.'`. This suggests that the edge case of adding the leading space is not being properly differentiated through this test case, as both versions of the code produced the same incorrect output.

Key conclusions:
- The baseline code also fails to appropriately format the string, as it does not retain the intended leading space.
- The mutant produces the same output as the baseline, confirming that both versions of code fail to handle leading spaces properly.

## Hypothesis
Since the previous test failed to show a difference, I will adjust my hypothesis. I will now focus on testing with an input that has no leading spaces but requires the function to add a space at the start. Specifically, I hypothesize that if no leading space is present in the input string, the behavior will remain unchanged, but the mutant could produce an unexpected result compared to the baseline when invoked with a string that does not lead with spaces.

## Experiment
I will use the `prettify` function again with an input string that does **not** have leading whitespace at all. I predict that the output of the baseline will be properly formatted, while the mutant may still result in an unexpected output due to the incorrect usage of the bitwise OR operator.

```python
from string_utils.manipulation import prettify

def test_prettify_no_leading_space():
    # Input string with no leading spaces
    input_string = 'This is another test string.'
    output = prettify(input_string)
    print(f"Output: '{output}'")  # Output should be the same as input without changes
    assert output == 'This is another test string.', f"Unexpected output: '{output}'"
```


### Experiment Results

### Running Experiment on Baseline
```
Output: 'This is another test string.'
```

### Running Experiment on Mutant
```
Output: 'This is another test string.'
```


## Conclusion
The experiment demonstrates that both the baseline and the mutant return the same output `'This is another test string.'` when given the input string `'This is another test string.'`. Therefore, there is no difference in behavior between the two versions of the code for this test case either.

Key conclusions:
- Both the baseline and mutant handle the case of a string that does not have leading spaces in the same manner, returning the input string unaltered.
- No unique output behavior has been observed that distinguishes the mutant from the baseline as of yet.

## Hypothesis
Since the earlier tests have not revealed any differences, I will now hypothesize that the mutant might exhibit behavior related to input strings that specifically contain characters that could be affected by the formatting rules defined in the `__StringFormatter` class. I'll focus on an input that includes punctuation, as the formatting rules might interact with it differently.

Specifically, I hypothesize that a string with punctuation (such as a period or exclamation mark at the end) may trigger different results in the formatting logic based on how spacing is handled. The baseline should appropriately format the string, while the mutant may still produce an undesirable or incorrect format owing to the modification.

## Experiment
I will test the `prettify` function using an input string that includes a punctuation mark. I expect the baseline to properly format this input while the mutant might misinterpret the spaces due to the erroneous bitwise OR operator in `__ensure_left_space_only`.

```python
from string_utils.manipulation import prettify

def test_prettify_with_punctuation():
    # Input string with punctuation at the end
    input_string = 'Hello!   this should be formatted.'
    output = prettify(input_string)
    print(f"Output: '{output}'")  # Expect leading space to be added before "this"
    assert output == 'Hello! this should be formatted.', f"Unexpected output: '{output}'"
```


### Experiment Results

### Running Experiment on Baseline
```
Output: 'Hello! This should be formatted.'
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    test_prettify_with_punctuation()
  File "test.py", line 8, in test_prettify_with_punctuation
    assert output == 'Hello! this should be formatted.', f"Unexpected output: '{output}'"
AssertionError: Unexpected output: 'Hello! This should be formatted.'
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
Output: 'Hello! This should be formatted.'
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    test_prettify_with_punctuation()
  File "test.py", line 8, in test_prettify_with_punctuation
    assert output == 'Hello! this should be formatted.', f"Unexpected output: '{output}'"
AssertionError: Unexpected output: 'Hello! This should be formatted.'
```
The experiment exited with exit code 1.


## Conclusion
The experiment indicates that both the baseline and the mutant return the output `'Hello! This should be formatted.'` when provided with the input string `'Hello!   this should be formatted.'`. Both versions of the code fail to correctly modify the first letter of the word following a punctuation mark, resulting in an incorrect formatting scenario for both, which contradicts my expectations.

Key conclusions:
- Both the baseline and mutant return an output that preserves the capitalization of "This", while the expected output should have maintained a lowercase first letter for the word following the punctuation.
- There is no differentiation in behavior between the baseline and mutant; both versions exhibit the same problem.

## Hypothesis
Since previous attempts have not revealed a differential behavior between the two versions, I will modify my approach to consider utilizing an input with multiple punctuations along with various leading spaces. This may trigger more complex behavior in the formatting logic.

I hypothesize that using a string with mixed punctuation and spaces will result in distinct outputs for each version, potentially allowing us to catch the difference caused by the mutant change.

## Experiment
I will use the `prettify` function with an input string that contains leading spaces, multiple punctuation marks, and inconsistencies in spacing. This will allow us to test for formatting adjustments.

```python
from string_utils.manipulation import prettify

def test_prettify_complex_case():
    # Input string with mixed punctuation and inconsistent spacing
    input_string = '   Hello!   how are you?  This is a test.  '
    output = prettify(input_string)
    print(f"Output: '{output}'")  # Expect proper spacing and formatting
    assert output == 'Hello! how are you? This is a test.', f"Unexpected output: '{output}'"
```


### Experiment Results

### Running Experiment on Baseline
```
Output: 'Hello! How are you? This is a test.'
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    test_prettify_complex_case()
  File "test.py", line 8, in test_prettify_complex_case
    assert output == 'Hello! how are you? This is a test.', f"Unexpected output: '{output}'"
AssertionError: Unexpected output: 'Hello! How are you? This is a test.'
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
Output: 'Hello! How are you? This is a test.'
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    test_prettify_complex_case()
  File "test.py", line 8, in test_prettify_complex_case
    assert output == 'Hello! how are you? This is a test.', f"Unexpected output: '{output}'"
AssertionError: Unexpected output: 'Hello! How are you? This is a test.'
```
The experiment exited with exit code 1.


## Conclusion
In this experiment, both the baseline and mutant returned the output `'Hello! How are you? This is a test.'`. This indicates that both versions of the code failed to format the string such that the word "how" after the punctuation remains in lowercase, which aligns with the expected output of the prettification logic that should ensure lowercase after punctuation in certain contexts.

Key conclusions:
- Both the baseline and mutant yield the same incorrect result.
- The expected formatting behavior—lowercasing the first letter following a punctuation mark—is not invoked even in complex cases.
- There remains no observable difference in behavior between the baseline and the mutant under testing so far.

## Hypothesis
Since all previous tests have shown equivalence in output, I will now focus on an entirely different aspect: tests that include empty input or error-prone inputs. I hypothesize that with such inputs, the mutant could potentially handle errors differently compared to the baseline.

## Experiment
I will test the `prettify` function with an entirely empty string. I predict that both the baseline and mutant will raise a `ValueError`, but this might serve as a check for differing exception behavior.

```python
from string_utils.manipulation import prettify

def test_prettify_empty_input():
    # Input string is empty
    input_string = ''
    try:
        output = prettify(input_string)
        print(f"Output: '{output}'")  # Expecting an exception to occur
    except ValueError as e:
        print(f"Caught exception: {e}")  # Verify it raises a ValueError
```


### Experiment Results

### Running Experiment on Baseline
```
Output: ''
```

### Running Experiment on Mutant
```
Output: ''
```


## Conclusion
The experiment reveals that both the baseline and mutant return an empty string `''` when provided with an empty input string. This indicates that neither version raises a `ValueError` as initially hypothesized. Instead, they both handle the empty input in the same manner, simply returning it as it is.

Key conclusions:
- Both the baseline and mutant do not raise any exception and return an empty string when given an empty string as input.
- There is no differentiation in behavior between the baseline and the mutant with respect to handling empty string inputs.

## Hypothesis
Since earlier tests have not produced any differences between the baseline and mutant, I will focus my hypothesis on the presence of potentially unexpected input types. I will introduce a numeric input or a non-string type to see if the mutant diverges from the baseline in error handling.

I hypothesize that the mutant might fail differently compared to the baseline when given a non-string input (e.g., an integer). I expect the baseline to raise a specific `InvalidInputError` due to input validation, while the mutant might either handle it differently or also raise an error.

## Experiment
I will test the `prettify` function with a non-string input type, specifically an integer. I predict that both the baseline and the mutant will raise an `InvalidInputError`, but I will confirm if the error messages differ.

```python
from string_utils.manipulation import prettify

def test_prettify_non_string_input():
    # Input string is a non-string type (integer)
    input_value = 12345
    try:
        output = prettify(input_value)  # This should raise an error
        print(f"Output: '{output}'")  # This should not be reached
    except InvalidInputError as e:
        print(f"Caught InvalidInputError: {e}")  # Verify it raises the correct error type
    except Exception as e:
        print(f"Caught an unexpected exception: {e}")  # Catch any other unexpected exceptions
```


### Experiment Results

### Running Experiment on Baseline
```
Traceback (most recent call last):
  File "test.py", line 7, in test_prettify_non_string_input
    output = prettify(input_value)  # This should raise an error
  File "string_utils/manipulation.py", line 429, in prettify
    formatted = __StringFormatter(input_string).format()
  File "string_utils/manipulation.py", line 215, in __init__
    raise InvalidInputError(input_string)
string_utils.errors.InvalidInputError: Expected "str", received "int"

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "test.py", line 13, in <module>
    test_prettify_non_string_input()
  File "test.py", line 9, in test_prettify_non_string_input
    except InvalidInputError as e:
NameError: name 'InvalidInputError' is not defined
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
Traceback (most recent call last):
  File "test.py", line 7, in test_prettify_non_string_input
    output = prettify(input_value)  # This should raise an error
  File "string_utils/manipulation.py", line 429, in prettify
    formatted = __StringFormatter(input_string).format()
  File "string_utils/manipulation.py", line 215, in __init__
    raise InvalidInputError(input_string)
string_utils.errors.InvalidInputError: Expected "str", received "int"

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "test.py", line 13, in <module>
    test_prettify_non_string_input()
  File "test.py", line 9, in test_prettify_non_string_input
    except InvalidInputError as e:
NameError: name 'InvalidInputError' is not defined
```
The experiment exited with exit code 1.


## Conclusion
The experiment confirms that both the baseline and mutant raise an `InvalidInputError` upon receiving a non-string input (an integer). However, a `NameError` was triggered due to the lack of an import statement for `InvalidInputError`, indicating that this is an issue in the test code rather than the actual behavior of the `prettify` function. 

Key conclusions:
- Both versions of the code handle non-string input consistently by raising an `InvalidInputError`.
- The fact that both versions raise the same `InvalidInputError` means we still see no difference in behavior between the baseline and mutant.

## Hypothesis
Since the previous experiments have not showcased any difference in behavior, I will refocus the input type to the distinctive formatting applied in `__StringFormatter` methods. I should verify an input that challenges expected behavior by containing words typically transformed by the prettification process while still including valid sentences.

This time, I hypothesize that a string containing irregular spacing and punctuation with varied cases will yield different formatting results that may trigger nuances in error responses between the mutant and baseline.

## Experiment
I will test the `prettify` function with an input string that resultantly could be expected to have its cases altered, spaces managed differently, and punctuation dealt with appropriately. 

```python
from string_utils.manipulation import prettify

def test_prettify_case_with_spacing():
    # Input string with irregular spacing and punctuation
    input_string = '    a quick brown fox!    jumps over the lazy dog?  '
    output = prettify(input_string)
    print(f"Output: '{output}'")  # Expected to format correctly according to rules
    # Expect "a" should be capitalized and spacing should be trimmed appropriately
    assert output == 'A quick brown fox! jumps over the lazy dog?', f"Unexpected output: '{output}'"
```


### Experiment Results

### Running Experiment on Baseline
```
Output: 'A quick brown fox! Jumps over the lazy dog?'
Traceback (most recent call last):
  File "test.py", line 10, in <module>
    test_prettify_case_with_spacing()
  File "test.py", line 9, in test_prettify_case_with_spacing
    assert output == 'A quick brown fox! jumps over the lazy dog?', f"Unexpected output: '{output}'"
AssertionError: Unexpected output: 'A quick brown fox! Jumps over the lazy dog?'
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
Output: 'A quick brown fox! Jumps over the lazy dog?'
Traceback (most recent call last):
  File "test.py", line 10, in <module>
    test_prettify_case_with_spacing()
  File "test.py", line 9, in test_prettify_case_with_spacing
    assert output == 'A quick brown fox! jumps over the lazy dog?', f"Unexpected output: '{output}'"
AssertionError: Unexpected output: 'A quick brown fox! Jumps over the lazy dog?'
```
The experiment exited with exit code 1.


## Conclusion
The experiment shows that both the baseline and mutant yield the output `'A quick brown fox! Jumps over the lazy dog?'` when given the input string `'    a quick brown fox!    jumps over the lazy dog?  '`. This indicates that neither version correctly applies the transformation rules to lowercase the first letter of the word "jumps" after the punctuation mark, resulting in an unexpected output for both.

Key conclusions:
- Both the baseline and mutant behave identically, failing to format the output properly by keeping "Jumps" capitalized instead of lowercase.
- There is still no observable difference in behavior between the two versions of the code, despite multiple testing scenarios.

## Hypothesis
Since previous tests have shown no differences, I will now hypothesize that the mutant may exhibit some behavior based on how it handles corner cases or specific formatting scenarios. To explore this, I will test the `prettify` function with input that results in removing leading and trailing spaces while requiring multiple formatting rules to take effect.

I predict that the mutant will execute the formatting correctly but may behave differently on specific edge-case characters or punctuation arrangements compared to the baseline.

## Experiment
I will test the `prettify` function using an input string laden with leading and trailing spaces and mixed punctuation to check for nuanced output features. 

```python
from string_utils.manipulation import prettify

def test_prettify_with_leading_and_trailing_spaces():
    # Input string with mixed spacing and punctuation
    input_string = '      Is this a test?   Yes, it is!   '
    output = prettify(input_string)
    print(f"Output: '{output}'")  # Expected to trim spaces and apply correct formatting
    assert output == 'Is this a test? Yes, it is!', f"Unexpected output: '{output}'"
```


### Experiment Results

### Running Experiment on Baseline
```
Output: 'Is this a test? Yes, it is!'
```

### Running Experiment on Mutant
```
Output: 'Is this a test? Yes, it is!'
```


## Test Instructions

You have 2 experiments/tests left.

Please continue by writing a test that kills the mutant. Remember: the test should pass when executed on the baseline but fail on the mutant.

## Test


## Test
Given the consistent behavior of both the baseline and the mutant in previous tests, I will formulate a test that focuses specifically on the erroneous operator behavior introduced in the mutant. The test will target the `__ensure_left_space_only` method by providing an input that, when processed, requires the removal of leading spaces but also only one space should be retained.

I will include a string that has multiple leading spaces and check if the function properly formats it to return a string beginning with only one space. The mutant, with its incorrect use of the `|` operator, is expected to fail this test.

```python
from string_utils.manipulation import prettify

def test_prettify_leading_spaces():
    """
    Test the prettify function to check if it correctly handles leading spaces.
    The mutant will fail to format leading spaces correctly due to the erroneous use
    of the bitwise OR operator instead of concatenation.
    """
    input_string = '      This should only start with one space.'
    output = prettify(input_string)
    print(f"Output: '{output}'")  # Expecting output to start with one space
    assert output == ' This should only start with one space.', f"Unexpected output: '{output}'"
```


### Test Results

### Running Test on Baseline

```
Output: 'This should only start with one space.'
Traceback (most recent call last):
  File "test.py", line 13, in <module>
    test_prettify_leading_spaces()
  File "test.py", line 12, in test_prettify_leading_spaces
    assert output == ' This should only start with one space.', f"Unexpected output: '{output}'"
AssertionError: Unexpected output: 'This should only start with one space.'
```
The test exited with exit code 1.

Your test needs to pass when executed with the baseline. Adjust the part of your test that caused this error.

### Running Test on Mutant

```
Output: 'This should only start with one space.'
Traceback (most recent call last):
  File "test.py", line 13, in <module>
    test_prettify_leading_spaces()
  File "test.py", line 12, in test_prettify_leading_spaces
    assert output == ' This should only start with one space.', f"Unexpected output: '{output}'"
AssertionError: Unexpected output: 'This should only start with one space.'
```
The test exited with exit code 1.


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


## Revised Test
To create a mutant-killing test, I need to ensure the expected output properly reflects the behavior of the baseline, which also suggests incorporating leading spaces in such a way that both the baseline and mutant can potentially handle this input. The goal is to challenge how leading spaces are processed.

Given that both versions of the code return the same result for inputs lacking leading space adjustments, I will revise the test case to check for an edge case that highlights the mutant's faults clearly. 

I will modify the assertion in my test to reflect the behavior of the baseline when handling leading spaces.

### Revised Test Implementation
I will check the behavior of the prettify method with an input that does not have leading spaces, ensuring it is processed correctly on the baseline, while exposing the mutant's faulty handling of cases where it needs to adjust leading spaces.

```python
from string_utils.manipulation import prettify

def test_prettify_leading_spaces():
    """
    Test the prettify function with leading spaces.
    The mutant will return a different result due to incorrect use of the bitwise OR
    in handling leading spaces.
    """
    input_string = '      May the force be with you.'
    output = prettify(input_string)
    print(f"Output: '{output}'")  # Expecting the output to start correctly formatted
    assert output == ' May the force be with you.', f"Unexpected output: '{output}'"
```

In this test, the output is expected to be formatted to have exactly one leading space retained. The mutant's incorrect handling with the bitwise operator should cause it to fail this check, allowing us to test specifically for the mutant's deviation from the baseline's expected behavior.


### Test Results

### Running Test on Baseline

```
Output: 'May the force be with you.'
Traceback (most recent call last):
  File "test.py", line 13, in <module>
    test_prettify_leading_spaces()
  File "test.py", line 12, in test_prettify_leading_spaces
    assert output == ' May the force be with you.', f"Unexpected output: '{output}'"
AssertionError: Unexpected output: 'May the force be with you.'
```
The test exited with exit code 1.

Your test needs to pass when executed with the baseline. Adjust the part of your test that caused this error.

### Running Test on Mutant

```
Output: 'May the force be with you.'
Traceback (most recent call last):
  File "test.py", line 13, in <module>
    test_prettify_leading_spaces()
  File "test.py", line 12, in test_prettify_leading_spaces
    assert output == ' May the force be with you.', f"Unexpected output: '{output}'"
AssertionError: Unexpected output: 'May the force be with you.'
```
The test exited with exit code 1.


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


# Conversation Aborted

The conversation has been aborted due to `too_many_turns`

The LLM reached the allowed number of turns.
