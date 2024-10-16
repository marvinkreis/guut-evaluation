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

```python string_utils/validation.py
001  # -*- coding: utf-8 -*-
002
003  # public api to export
004  __all__ = [
005      'is_string',
006      'is_full_string',
007      'is_number',
008      'is_integer',
009      'is_decimal',
010      'is_url',
011      'is_email',
012      'is_credit_card',
013      'is_camel_case',
014      'is_snake_case',
015      'is_json',
016      'is_uuid',
017      'is_ip_v4',
018      'is_ip_v6',
019      'is_ip',
020      'is_isbn_10',
021      'is_isbn_13',
022      'is_isbn',
023      'is_palindrome',
024      'is_pangram',
025      'is_isogram',
026      'is_slug',
027      'contains_html',
028      'words_count',
029  ]
030
031  import json
032  import string
033  from typing import Any, Optional, List
034
035  from ._regex import *
036  from .errors import InvalidInputError
037
038
039  # PRIVATE API
040
041
042  class __ISBNChecker:
043      def __init__(self, input_string: str, normalize: bool = True):
044          if not is_string(input_string):
045              raise InvalidInputError(input_string)
046
047          self.input_string = input_string.replace('-', '') if normalize else input_string
048
049      def is_isbn_13(self) -> bool:
050          if len(self.input_string) == 13:
051              product = 0
052
053              try:
054                  for index, digit in enumerate(self.input_string):
055                      weight = 1 if (index % 2 == 0) else 3
056                      product += int(digit) * weight
057
058                  return product % 10 == 0
059
060              except ValueError:
061                  pass
062
063          return False
064
065      def is_isbn_10(self) -> bool:
066          if len(self.input_string) == 10:
067              product = 0
068
069              try:
070                  for index, digit in enumerate(self.input_string):
071                      product += int(digit) * (index + 1)
072
073                  return product % 11 == 0
074
075              except ValueError:
076                  pass
077
078          return False
079
080
081  # PUBLIC API
082
083  def is_string(obj: Any) -> bool:
084      """
085      Checks if an object is a string.
086
087      *Example:*
088
089      >>> is_string('foo') # returns true
090      >>> is_string(b'foo') # returns false
091
092      :param obj: Object to test.
093      :return: True if string, false otherwise.
094      """
095      return isinstance(obj, str)
096
097
098  def is_full_string(input_string: Any) -> bool:
099      """
100      Check if a string is not empty (it must contains at least one non space character).
101
102      *Examples:*
103
104      >>> is_full_string(None) # returns false
105      >>> is_full_string('') # returns false
106      >>> is_full_string(' ') # returns false
107      >>> is_full_string('hello') # returns true
108
109      :param input_string: String to check.
110      :type input_string: str
111      :return: True if not empty, false otherwise.
112      """
113      return is_string(input_string) and input_string.strip() != ''
114
115
116  def is_number(input_string: str) -> bool:
117      """
118      Checks if a string is a valid number.
119
120      The number can be a signed (eg: +1, -2, -3.3) or unsigned (eg: 1, 2, 3.3) integer or double
121      or use the "scientific notation" (eg: 1e5).
122
123      *Examples:*
124
125      >>> is_number('42') # returns true
126      >>> is_number('19.99') # returns true
127      >>> is_number('-9.12') # returns true
128      >>> is_number('1e3') # returns true
129      >>> is_number('1 2 3') # returns false
130
131      :param input_string: String to check
132      :type input_string: str
133      :return: True if the string represents a number, false otherwise
134      """
135      if not isinstance(input_string, str):
136          raise InvalidInputError(input_string)
137
138      return NUMBER_RE.match(input_string) is not None
139
140
141  def is_integer(input_string: str) -> bool:
142      """
143      Checks whether the given string represents an integer or not.
144
145      An integer may be signed or unsigned or use a "scientific notation".
146
147      *Examples:*
148
149      >>> is_integer('42') # returns true
150      >>> is_integer('42.0') # returns false
151
152      :param input_string: String to check
153      :type input_string: str
154      :return: True if integer, false otherwise
155      """
156      return is_number(input_string) and '.' not in input_string
157
158
159  def is_decimal(input_string: str) -> bool:
160      """
161      Checks whether the given string represents a decimal or not.
162
163      A decimal may be signed or unsigned or use a "scientific notation".
164
165      >>> is_decimal('42.0') # returns true
166      >>> is_decimal('42') # returns false
167
168      :param input_string: String to check
169      :type input_string: str
170      :return: True if integer, false otherwise
171      """
172      return is_number(input_string) and '.' in input_string
173
174
175  # Full url example:
176  # scheme://username:password@www.domain.com:8042/folder/subfolder/file.extension?param=value&param2=value2#hash
177  def is_url(input_string: Any, allowed_schemes: Optional[List[str]] = None) -> bool:
178      """
179      Check if a string is a valid url.
180
181      *Examples:*
182
183      >>> is_url('http://www.mysite.com') # returns true
184      >>> is_url('https://mysite.com') # returns true
185      >>> is_url('.mysite.com') # returns false
186
187      :param input_string: String to check.
188      :type input_string: str
189      :param allowed_schemes: List of valid schemes ('http', 'https', 'ftp'...). Default to None (any scheme is valid).
190      :type allowed_schemes: Optional[List[str]]
191      :return: True if url, false otherwise
192      """
193      if not is_full_string(input_string):
194          return False
195
196      valid = URL_RE.match(input_string) is not None
197
198      if allowed_schemes:
199          return valid and any([input_string.startswith(s) for s in allowed_schemes])
200
201      return valid
202
203
204  def is_email(input_string: Any) -> bool:
205      """
206      Check if a string is a valid email.
207
208      Reference: https://tools.ietf.org/html/rfc3696#section-3
209
210      *Examples:*
211
212      >>> is_email('my.email@the-provider.com') # returns true
213      >>> is_email('@gmail.com') # returns false
214
215      :param input_string: String to check.
216      :type input_string: str
217      :return: True if email, false otherwise.
218      """
219      # first simple "pre check": it must be a non empty string with max len 320 and cannot start with a dot
220      if not is_full_string(input_string) or len(input_string) > 320 or input_string.startswith('.'):
221          return False
222
223      try:
224          # we expect 2 tokens, one before "@" and one after, otherwise we have an exception and the email is not valid
225          head, tail = input_string.split('@')
226
227          # head's size must be <= 64, tail <= 255, head must not start with a dot or contain multiple consecutive dots
228          if len(head) > 64 or len(tail) > 255 or head.endswith('.') or ('..' in head):
229              return False
230
231          # removes escaped spaces, so that later on the test regex will accept the string
232          head = head.replace('\\ ', '')
233          if head.startswith('"') and head.endswith('"'):
234              head = head.replace(' ', '')[1:-1]
235
236          return EMAIL_RE.match(head + '@' + tail) is not None
237
238      except ValueError:
239          # borderline case in which we have multiple "@" signs but the head part is correctly escaped
240          if ESCAPED_AT_SIGN.search(input_string) is not None:
241              # replace "@" with "a" in the head
242              return is_email(ESCAPED_AT_SIGN.sub('a', input_string))
243
244          return False
245
246
247  def is_credit_card(input_string: Any, card_type: str = None) -> bool:
248      """
249      Checks if a string is a valid credit card number.
250      If card type is provided then it checks against that specific type only,
251      otherwise any known credit card number will be accepted.
252
253      Supported card types are the following:
254
255      - VISA
256      - MASTERCARD
257      - AMERICAN_EXPRESS
258      - DINERS_CLUB
259      - DISCOVER
260      - JCB
261
262      :param input_string: String to check.
263      :type input_string: str
264      :param card_type: Card type. Default to None (any card).
265      :type card_type: str
266
267      :return: True if credit card, false otherwise.
268      """
269      if not is_full_string(input_string):
270          return False
271
272      if card_type:
273          if card_type not in CREDIT_CARDS:
274              raise KeyError(
275                  'Invalid card type "{}". Valid types are: {}'.format(card_type, ', '.join(CREDIT_CARDS.keys()))
276              )
277          return CREDIT_CARDS[card_type].match(input_string) is not None
278
279      for c in CREDIT_CARDS:
280          if CREDIT_CARDS[c].match(input_string) is not None:
281              return True
282
283      return False
284
285
286  def is_camel_case(input_string: Any) -> bool:
287      """
288      Checks if a string is formatted as camel case.
289
290      A string is considered camel case when:
291
292      - it's composed only by letters ([a-zA-Z]) and optionally numbers ([0-9])
293      - it contains both lowercase and uppercase letters
294      - it does not start with a number
295
296      *Examples:*
297
298      >>> is_camel_case('MyString') # returns true
299      >>> is_camel_case('mystring') # returns false
300
301      :param input_string: String to test.
302      :type input_string: str
303      :return: True for a camel case string, false otherwise.
304      """
305      return is_full_string(input_string) and CAMEL_CASE_TEST_RE.match(input_string) is not None
306
307
308  def is_snake_case(input_string: Any, separator: str = '_') -> bool:
309      """
310      Checks if a string is formatted as "snake case".
311
312      A string is considered snake case when:
313
314      - it's composed only by lowercase/uppercase letters and digits
315      - it contains at least one underscore (or provided separator)
316      - it does not start with a number
317
318      *Examples:*
319
320      >>> is_snake_case('foo_bar_baz') # returns true
321      >>> is_snake_case('foo') # returns false
322
323      :param input_string: String to test.
324      :type input_string: str
325      :param separator: String to use as separator.
326      :type separator: str
327      :return: True for a snake case string, false otherwise.
328      """
329      if is_full_string(input_string):
330          re_map = {
331              '_': SNAKE_CASE_TEST_RE,
332              '-': SNAKE_CASE_TEST_DASH_RE
333          }
334          re_template = r'([a-z]+\d*{sign}[a-z\d{sign}]*|{sign}+[a-z\d]+[a-z\d{sign}]*)'
335          r = re_map.get(
336              separator,
337              re.compile(re_template.format(sign=re.escape(separator)), re.IGNORECASE)
338          )
339
340          return r.match(input_string) is not None
341
342      return False
343
344
345  def is_json(input_string: Any) -> bool:
346      """
347      Check if a string is a valid json.
348
349      *Examples:*
350
351      >>> is_json('{"name": "Peter"}') # returns true
352      >>> is_json('[1, 2, 3]') # returns true
353      >>> is_json('{nope}') # returns false
354
355      :param input_string: String to check.
356      :type input_string: str
357      :return: True if json, false otherwise
358      """
359      if is_full_string(input_string) and JSON_WRAPPER_RE.match(input_string) is not None:
360          try:
361              return isinstance(json.loads(input_string), (dict, list))
362          except (TypeError, ValueError, OverflowError):
363              pass
364
365      return False
366
367
368  def is_uuid(input_string: Any, allow_hex: bool = False) -> bool:
369      """
370      Check if a string is a valid UUID.
371
372      *Example:*
373
374      >>> is_uuid('6f8aa2f9-686c-4ac3-8766-5712354a04cf') # returns true
375      >>> is_uuid('6f8aa2f9686c4ac387665712354a04cf') # returns false
376      >>> is_uuid('6f8aa2f9686c4ac387665712354a04cf', allow_hex=True) # returns true
377
378      :param input_string: String to check.
379      :type input_string: str
380      :param allow_hex: True to allow UUID hex representation as valid, false otherwise (default)
381      :type allow_hex: bool
382      :return: True if UUID, false otherwise
383      """
384      # string casting is used to allow UUID itself as input data type
385      s = str(input_string)
386
387      if allow_hex:
388          return UUID_HEX_OK_RE.match(s) is not None
389
390      return UUID_RE.match(s) is not None
391
392
393  def is_ip_v4(input_string: Any) -> bool:
394      """
395      Checks if a string is a valid ip v4.
396
397      *Examples:*
398
399      >>> is_ip_v4('255.200.100.75') # returns true
400      >>> is_ip_v4('nope') # returns false (not an ip)
401      >>> is_ip_v4('255.200.100.999') # returns false (999 is out of range)
402
403      :param input_string: String to check.
404      :type input_string: str
405      :return: True if an ip v4, false otherwise.
406      """
407      if not is_full_string(input_string) or SHALLOW_IP_V4_RE.match(input_string) is None:
408          return False
409
410      # checks that each entry in the ip is in the valid range (0 to 255)
411      for token in input_string.split('.'):
412          if not (0 <= int(token) <= 255):
413              return False
414
415      return True
416
417
418  def is_ip_v6(input_string: Any) -> bool:
419      """
420      Checks if a string is a valid ip v6.
421
422      *Examples:*
423
424      >>> is_ip_v6('2001:db8:85a3:0000:0000:8a2e:370:7334') # returns true
425      >>> is_ip_v6('2001:db8:85a3:0000:0000:8a2e:370:?') # returns false (invalid "?")
426
427      :param input_string: String to check.
428      :type input_string: str
429      :return: True if a v6 ip, false otherwise.
430      """
431      return is_full_string(input_string) and IP_V6_RE.match(input_string) is not None
432
433
434  def is_ip(input_string: Any) -> bool:
435      """
436      Checks if a string is a valid ip (either v4 or v6).
437
438      *Examples:*
439
440      >>> is_ip('255.200.100.75') # returns true
441      >>> is_ip('2001:db8:85a3:0000:0000:8a2e:370:7334') # returns true
442      >>> is_ip('1.2.3') # returns false
443
444      :param input_string: String to check.
445      :type input_string: str
446      :return: True if an ip, false otherwise.
447      """
448      return is_ip_v6(input_string) or is_ip_v4(input_string)
449
450
451  def is_palindrome(input_string: Any, ignore_spaces: bool = False, ignore_case: bool = False) -> bool:
452      """
453      Checks if the string is a palindrome (https://en.wikipedia.org/wiki/Palindrome).
454
455      *Examples:*
456
457      >>> is_palindrome('LOL') # returns true
458      >>> is_palindrome('Lol') # returns false
459      >>> is_palindrome('Lol', ignore_case=True) # returns true
460      >>> is_palindrome('ROTFL') # returns false
461
462      :param input_string: String to check.
463      :type input_string: str
464      :param ignore_spaces: False if white spaces matter (default), true otherwise.
465      :type ignore_spaces: bool
466      :param ignore_case: False if char case matters (default), true otherwise.
467      :type ignore_case: bool
468      :return: True if the string is a palindrome (like "otto", or "i topi non avevano nipoti" if strict=False),\
469      False otherwise
470      """
471      if not is_full_string(input_string):
472          return False
473
474      if ignore_spaces:
475          input_string = SPACES_RE.sub('', input_string)
476
477      string_len = len(input_string)
478
479      # Traverse the string one char at step, and for each step compares the
480      # "head_char" (the one on the left of the string) to the "tail_char" (the one on the right).
481      # In this way we avoid to manipulate the whole string in advance if not necessary and provide a faster
482      # algorithm which can scale very well for long strings.
483      for index in range(string_len):
484          head_char = input_string[index]
485          tail_char = input_string[string_len - index - 1]
486
487          if ignore_case:
488              head_char = head_char.lower()
489              tail_char = tail_char.lower()
490
491          if head_char != tail_char:
492              return False
493
494      return True
495
496
497  def is_pangram(input_string: Any) -> bool:
498      """
499      Checks if the string is a pangram (https://en.wikipedia.org/wiki/Pangram).
500
501      *Examples:*
502
503      >>> is_pangram('The quick brown fox jumps over the lazy dog') # returns true
504      >>> is_pangram('hello world') # returns false
505
506      :param input_string: String to check.
507      :type input_string: str
508      :return: True if the string is a pangram, False otherwise.
509      """
510      if not is_full_string(input_string):
511          return False
512
513      return set(SPACES_RE.sub('', input_string)).issuperset(set(string.ascii_lowercase))
514
515
516  def is_isogram(input_string: Any) -> bool:
517      """
518      Checks if the string is an isogram (https://en.wikipedia.org/wiki/Isogram).
519
520      *Examples:*
521
522      >>> is_isogram('dermatoglyphics') # returns true
523      >>> is_isogram('hello') # returns false
524
525      :param input_string: String to check.
526      :type input_string: str
527      :return: True if isogram, false otherwise.
528      """
529      return is_full_string(input_string) and len(set(input_string)) == len(input_string)
530
531
532  def is_slug(input_string: Any, separator: str = '-') -> bool:
533      """
534      Checks if a given string is a slug (as created by `slugify()`).
535
536      *Examples:*
537
538      >>> is_slug('my-blog-post-title') # returns true
539      >>> is_slug('My blog post title') # returns false
540
541      :param input_string: String to check.
542      :type input_string: str
543      :param separator: Join sign used by the slug.
544      :type separator: str
545      :return: True if slug, false otherwise.
546      """
547      if not is_full_string(input_string):
548          return False
549
550      rex = r'^([a-z\d]+' + re.escape(separator) + r'*?)*[a-z\d]$'
551
552      return re.match(rex, input_string) is not None
553
554
555  def contains_html(input_string: str) -> bool:
556      """
557      Checks if the given string contains HTML/XML tags.
558
559      By design, this function matches ANY type of tag, so don't expect to use it
560      as an HTML validator, its goal is to detect "malicious" or undesired tags in the text.
561
562      *Examples:*
563
564      >>> contains_html('my string is <strong>bold</strong>') # returns true
565      >>> contains_html('my string is not bold') # returns false
566
567      :param input_string: Text to check
568      :type input_string: str
569      :return: True if string contains html, false otherwise.
570      """
571      if not is_string(input_string):
572          raise InvalidInputError(input_string)
573
574      return HTML_RE.search(input_string) is not None
575
576
577  def words_count(input_string: str) -> int:
578      """
579      Returns the number of words contained into the given string.
580
581      This method is smart, it does consider only sequence of one or more letter and/or numbers
582      as "words", so a string like this: "! @ # % ... []" will return zero!
583      Moreover it is aware of punctuation, so the count for a string like "one,two,three.stop"
584      will be 4 not 1 (even if there are no spaces in the string).
585
586      *Examples:*
587
588      >>> words_count('hello world') # returns 2
589      >>> words_count('one,two,three.stop') # returns 4
590
591      :param input_string: String to check.
592      :type input_string: str
593      :return: Number of words.
594      """
595      if not is_string(input_string):
596          raise InvalidInputError(input_string)
597
598      return len(WORDS_COUNT_RE.findall(input_string))
599
600
601  def is_isbn_10(input_string: str, normalize: bool = True) -> bool:
602      """
603      Checks if the given string represents a valid ISBN 10 (International Standard Book Number).
604      By default hyphens in the string are ignored, so digits can be separated in different ways, by calling this
605      function with `normalize=False` only digit-only strings will pass the validation.
606
607      *Examples:*
608
609      >>> is_isbn_10('1506715214') # returns true
610      >>> is_isbn_10('150-6715214') # returns true
611      >>> is_isbn_10('150-6715214', normalize=False) # returns false
612
613      :param input_string: String to check.
614      :param normalize: True to ignore hyphens ("-") in the string (default), false otherwise.
615      :return: True if valid ISBN 10, false otherwise.
616      """
617      checker = __ISBNChecker(input_string, normalize)
618      return checker.is_isbn_10()
619
620
621  def is_isbn_13(input_string: str, normalize: bool = True) -> bool:
622      """
623      Checks if the given string represents a valid ISBN 13 (International Standard Book Number).
624      By default hyphens in the string are ignored, so digits can be separated in different ways, by calling this
625      function with `normalize=False` only digit-only strings will pass the validation.
626
627      *Examples:*
628
629      >>> is_isbn_13('9780312498580') # returns true
630      >>> is_isbn_13('978-0312498580') # returns true
631      >>> is_isbn_13('978-0312498580', normalize=False) # returns false
632
633      :param input_string: String to check.
634      :param normalize: True to ignore hyphens ("-") in the string (default), false otherwise.
635      :return: True if valid ISBN 13, false otherwise.
636      """
637      checker = __ISBNChecker(input_string, normalize)
638      return checker.is_isbn_13()
639
640
641  def is_isbn(input_string: str, normalize: bool = True) -> bool:
642      """
643      Checks if the given string represents a valid ISBN (International Standard Book Number).
644      By default hyphens in the string are ignored, so digits can be separated in different ways, by calling this
645      function with `normalize=False` only digit-only strings will pass the validation.
646
647      *Examples:*
648
649      >>> is_isbn('9780312498580') # returns true
650      >>> is_isbn('1506715214') # returns true
651
652      :param input_string: String to check.
653      :param normalize: True to ignore hyphens ("-") in the string (default), false otherwise.
654      :return: True if valid ISBN (10 or 13), false otherwise.
655      """
656      checker = __ISBNChecker(input_string, normalize)
657      return checker.is_isbn_13() or checker.is_isbn_10()
```

```diff mutant.diff
diff --git a/string_utils/validation.py b/mutant/string_utils/validation.py
index 9124cbb..2a26994 100644
--- a/string_utils/validation.py
+++ b/mutant/string_utils/validation.py
@@ -53,11 +53,11 @@ class __ISBNChecker:
             try:
                 for index, digit in enumerate(self.input_string):
                     weight = 1 if (index % 2 == 0) else 3
                     product += int(digit) * weight
 
-                return product % 10 == 0
+                return product % 10 <= 0
 
             except ValueError:
                 pass
 
         return False
```

# Debugging


## Hypothesis
The mutant changes the logic in the `is_isbn_13` method from returning `True` when the ISBN check is valid (i.e., the product modulo 10 equals 0) to returning `True` if the product modulo 10 is less than or equal to 0. This means that when computing the validity of an ISBN-13 number, the mutant may incorrectly validate more strings (including invalid ones) as valid ISBN-13 numbers. I hypothesize that this mutation will incorrectly validate valid ISBN-13 strings if their product is not a multiple of 10, such as '9780312498581'.

## Experiment
I will call `is_isbn_13` with the valid ISBN-13 `9780312498580`, which should return `True` for the baseline. For the test against the mutant, I will also call `is_isbn_13` with an invalid ISBN-13 number `9780312498581`, which should return `False` in both versions, prompting the mutant to mishandle this input. I predict the baseline will correctly identify that `9780312498580` is valid and `9780312498581` is invalid, while the mutant will incorrectly validate `9780312498581`.

```python
from string_utils.validation import is_isbn_13

def test_is_isbn_13():
    # Check a valid ISBN-13 number
    valid_isbn = '9780312498580'
    invalid_isbn = '9780312498581'
    
    valid_output = is_isbn_13(valid_isbn)
    invalid_output = is_isbn_13(invalid_isbn)
    
    print(f"Valid ISBN Output: {valid_output}")  # Expecting True
    print(f"Invalid ISBN Output: {invalid_output}")  # Expecting False
    assert valid_output == True, f"Expected True, got {valid_output}"
    assert invalid_output == False, f"Expected False, got {invalid_output}"
```


### Experiment Results

### Running Experiment on Baseline
```
Valid ISBN Output: True
Invalid ISBN Output: False
```

### Running Experiment on Mutant
```
Valid ISBN Output: True
Invalid ISBN Output: False
```


## Conclusion
The experiment shows that both the baseline and the mutant produce the same output for the valid ISBN-13 (`9780312498580`), correctly returning `True`. However, the invalid ISBN-13 (`9780312498581`) also returns `False` for both versions. This contradicts the hypothesis that the mutant would incorrectly validate invalid ISBN-13 numbers.

Key conclusions:
- Both baseline and mutant correctly return `True` for a valid ISBN-13.
- Both versions correctly return `False` for an invalid ISBN-13.

Since this experiment did not reveal any differences in output between the baseline and the mutant, I will refine my hypothesis and experiment by focusing on an edge case which is a valid ISBN-13 number but has a product not equal to 0 when taking modulo 10.

## Hypothesis
The mutation could lead to incorrect results when a valid ISBN-13 number has a `product % 10` that is non-zero, potentially allowing it to falsely validate some invalid ISBN-13 numbers. I will now test a valid ISBN-13 `9780312498580` which has a valid product modulo 10 equal to zero and an invalid ISBN-13 number where `product % 10` is non-zero, such as `9780312498581`.

## Experiment
I will call `is_isbn_13` with both an invalid ISBN-13 `9780312498581`, which we expect to be evaluated as `False` by the baseline, and `9780312498582`, which has the same digits but is an invalid ISBN-13. I predict that the baseline will return `False`, while the mutant might incorrectly return `True`.

```python
from string_utils.validation import is_isbn_13

def test_is_isbn_13_edge_case():
    # Invalid ISBN-13 that is close to a valid one
    invalid_isbn1 = '9780312498581'  # Valid format but should be invalid
    invalid_isbn2 = '9780312498582'  # Same format, also should be invalid
    
    invalid_output1 = is_isbn_13(invalid_isbn1)
    invalid_output2 = is_isbn_13(invalid_isbn2)
    
    print(f"Invalid ISBN 1 Output: {invalid_output1}")  # Expecting False
    print(f"Invalid ISBN 2 Output: {invalid_output2}")  # Expecting False
    
    assert invalid_output1 == False, f"Expected False for {invalid_isbn1}, got {invalid_output1}"
    assert invalid_output2 == False, f"Expected False for {invalid_isbn2}, got {invalid_output2}"
```


### Experiment Results

### Running Experiment on Baseline
```
Invalid ISBN 1 Output: False
Invalid ISBN 2 Output: False
```

### Running Experiment on Mutant
```
Invalid ISBN 1 Output: False
Invalid ISBN 2 Output: False
```


## Conclusion
The results of this experiment also show that both the baseline and the mutant correctly return `False` for both invalid ISBN-13 numbers (`9780312498581` and `9780312498582`). This again contradicts my hypothesis that the mutant would validate an invalid ISBN-13 incorrectly.

Key conclusions:
- Both versions correctly identify the invalid ISBN-13 numbers as `False`.

Given that the previous tests have not revealed any differences, I will adapt the hypothesis to consider ISBN-13 numbers that are not formatted properly, but are recognized as valid by the algorithm.

## Hypothesis
I now hypothesize that the mutant might allow certain malformed ISBN-13 strings that do not follow the traditional structure (e.g., a string that is too short or contains letters). For example, an input such as `978-0-306-40615-7` which is a valid ISBN-13 format but might get mismatched due to parsing issues in the mutant's check.

## Experiment
I will run `is_isbn_13` with a poorly formed ISBN value such as `978x0312498580` that includes a letter instead of a valid digit, which clearly should return `False`. I predict that this might expose the mutant's flaw if it does not check the string properly.

```python
from string_utils.validation import is_isbn_13

def test_is_isbn_13_malformed():
    # Malformed ISBN-13 with a letter
    malformed_isbn = '978x0312498580'  # Should be invalid due to 'x'
    
    malformed_output = is_isbn_13(malformed_isbn)
    
    print(f"Malformed ISBN Output: {malformed_output}")  # Expecting False
    assert malformed_output == False, f"Expected False, got {malformed_output}"
```


### Experiment Results

### Running Experiment on Baseline
```
Malformed ISBN Output: False
```

### Running Experiment on Mutant
```
Malformed ISBN Output: False
```


## Conclusion
Both the baseline and the mutant correctly returned `False` for the malformed ISBN-13 input `978x0312498580`. This indicates that the mutant is handling malformed inputs consistently with the baseline, which again does not support my hypothesis.

Key conclusions:
- Both versions correctly identify the malformed ISBN-13 as invalid.

Since I have not yet produced any differences with the mutant, I will try a more specific and controlled approach by testing input values that are right at the edge of validity for ISBN-13 checks.

## Hypothesis
I hypothesize that the mutant could be exposed when we provide well-structured but incorrect values, specifically the last digit which determines the validity. The calculation on the product could lead to incorrect results if it doesn't check for the final condition properly. For instance, the validity of `9780312498582` should return `False` for the baseline and potentially `True` for the mutant due to its flawed logic.

## Experiment
I will call `is_isbn_13` with `9780312498582`, which is close to a valid ISBN-13 but is indeed invalid. I expect the baseline to return `False`, while the mutant could incorrectly return `True`.

```python
from string_utils.validation import is_isbn_13

def test_is_isbn_13_final_digit_validity():
    # Invalid ISBN-13 but very close to valid
    close_to_valid_isbn = '9780312498582'  # Should be invalid
    
    close_to_valid_output = is_isbn_13(close_to_valid_isbn)
    
    print(f"Close to Valid ISBN Output: {close_to_valid_output}")  # Expecting False
    assert close_to_valid_output == False, f"Expected False, got {close_to_valid_output}"
```


### Experiment Results

### Running Experiment on Baseline
```
Close to Valid ISBN Output: False
```

### Running Experiment on Mutant
```
Close to Valid ISBN Output: False
```


## Conclusion
Once again, the results indicate that both the baseline and the mutant correctly return `False` for the input `9780312498582`, which is invalid. This means that the mutation does not expose any differences in behavior for this test either.

Key conclusions:
- Both versions correctly reject the close-to-valid ISBN-13 as `False`.

Since the previous hypothesis did not lead to any distinction, I will change my approach. I will explore the main part of the code that is causing the mutation, targeting the normalization aspect and providing a raw ISBN input with hyphens that should be rejected by the baseline but could be improperly handled by the mutant.

## Hypothesis
I hypothesize that the mutant will improperly handle ISBNS that contain hyphens. I will test a valid ISBN-13 input like `978-0-306-40615-7`, which should be treated correctly only if normalized.

## Experiment
I will call `is_isbn_13` with the properly formatted ISBN-13 with hyphens and check whether it recognizes the number correctly. I predict that the baseline will return `True`, while the mutant will improperly return `False`, due to logic errors around the handling of hyphens and their effects on the computation.

```python
from string_utils.validation import is_isbn_13

def test_is_isbn_13_with_hyphens():
    # Valid ISBN-13 with hyphens
    isbn_with_hyphens = '978-0-306-40615-7'  # Should be valid after normalization
    
    output = is_isbn_13(isbn_with_hyphens)
    
    print(f"ISBN with Hyphens Output: {output}")  # Expecting True
    assert output == True, f"Expected True, got {output}"
```


### Experiment Results

### Running Experiment on Baseline
```
ISBN with Hyphens Output: True
```

### Running Experiment on Mutant
```
ISBN with Hyphens Output: True
```


## Conclusion
The experiment results show that both the baseline and the mutant correctly return `True` for the valid ISBN-13 input with hyphens (`978-0-306-40615-7`). This indicates that the mutant is still behaving in accordance with the baseline.

Key conclusions:
- Both versions correctly handle the ISBN-13 with hyphens, returning `True`.

Given the lack of differences so far, I will take a different approach by testing a case with an invalid structure that also has valid digits. I aim to explore how the mutant addresses numbers with varying lengths, specifically with 10-character ISBN-13s, where the last character does not influence the validity.

## Hypothesis
I hypothesize that the mutant will fail to reject an ISBN-13 string with an incorrect number of characters, specifically if concerns arise from a variant of digits at the end. I will test an invalid 13-character string `978031249858`, which is one character short.

## Experiment
I will call `is_isbn_13` with the invalid 12-character string and check whether it recognizes this as invalid. I predict that the baseline will return `False`, while the mutated code may incorrectly handle it due to its changed logic concerning the validation check on products.

```python
from string_utils.validation import is_isbn_13

def test_is_isbn_13_twelve_characters():
    # Invalid input with too few characters (12 instead of 13)
    invalid_isbn = '978031249858'  # Should be invalid due to length
    
    output = is_isbn_13(invalid_isbn)
    
    print(f"Twelve Character ISBN Output: {output}")  # Expecting False
    assert output == False, f"Expected False, got {output}"
```


### Experiment Results

### Running Experiment on Baseline
```
Twelve Character ISBN Output: False
```

### Running Experiment on Mutant
```
Twelve Character ISBN Output: False
```


## Conclusion
The results show that both the baseline and the mutant correctly identify the invalid ISBN-13 input with only 12 characters (`978031249858`) as `False`. This reiterates that the mutant's faulty logic does not negatively affect its ability to reject this particular case.

Key conclusions:
- Both versions accurately reject the 12-character input ISBN-13 as invalid.

After multiple experiments, there have been no notable instances where the mutant displayed different behavior compared to the baseline. I'll summarize my findings to explore whether the mutant is indeed equivalent to the baseline.

## Equivalent Mutant
The mutant changes the return condition from `product % 10 == 0` to `product % 10 <= 0`. This doesn't seem to produce a different behavior under the inputs tested. Valid and invalid inputs have been tested, and both the baseline and the mutant exhibit consistent behaviors in rejecting or accepting ISBN-13 strings.

Although my initial hypotheses targeted logic flaws, the tests show that both implementations handle cases similarly. Consequently, I believe the mutant is equivalent because it does not introduce a meaningful distinction in behavior that can be detected by the tests executed. 

Given this, I will not write a separate test as there are no significant differences between the mutant and the baseline for the inputs tested. The mutations do not result in a failure or exception under conditions which might otherwise reveal discrepancies.


You claimed that the mutant as equivalent.

Now, try other experiments to check if you can prove yourself wrong. Try new approaches or new inputs you haven't tried yet. See if you can maybe find a difference between the **Baseline** and the **Mutant** anyways.


## New Hypothesis
Given the nature of the mutation, I will now focus on testing ISBN-13 inputs where the digits sum varies significantly. Specifically, I will check inputs where the last digit is malformed but follows the structure conventions of an ISBN-13 format. I suspect that this could lead the mutant to produce an incorrect result due to its revised logic regarding the return condition.

### Inputs to Test
1. A valid ISBN-13 (`9780312498580`), which should return `True` in both versions.
2. An invalid ISBN-13 with a malformed last character (not a digit) or differing digit structure. Perhaps `978031249858Z`, where `Z` is not a valid digit.

## Experiment
I will call `is_isbn_13` with the malformed input (`978031249858Z`). I predict that while the baseline will return `False` (since it checks for numeric values), the mutant may incorrectly return `True`.

```python
from string_utils.validation import is_isbn_13

def test_is_isbn_13_malformed_last_character():
    # Invalid ISBN-13 with a non-numeric character
    malformed_isbn = '978031249858Z'  # Should be invalid due to non-numeric character

    output = is_isbn_13(malformed_isbn)

    print(f"Malformed Last Character ISBN Output: {output}")  # Expecting False
    assert output == False, f"Expected False, got {output}"
```

### Additional Experiment
I will also try calling `is_isbn_13` using an entirely incorrect pattern but valid structure, such as `978-0-306-4A615-7`. This last character `A` is inappropriate and should show variance, yielding a different behavior between mutant and baseline.

```python
def test_is_isbn_13_invalid_with_alpha():
    # Invalid ISBN-13 with a letter
    invalid_isbn_with_letter = '978-0-306-4A615-7'  # Should be invalid

    output = is_isbn_13(invalid_isbn_with_letter)

    print(f"Invalid ISBN with Letter Output: {output}")  # Expecting False
    assert output == False, f"Expected False, got {output}"
```

Let's run these experiments.


### Experiment Results

### Running Experiment on Baseline
```
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    test_is_isbn_13_invalid_with_alpha()
  File "test.py", line 5, in test_is_isbn_13_invalid_with_alpha
    output = is_isbn_13(invalid_isbn_with_letter)
NameError: name 'is_isbn_13' is not defined
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    test_is_isbn_13_invalid_with_alpha()
  File "test.py", line 5, in test_is_isbn_13_invalid_with_alpha
    output = is_isbn_13(invalid_isbn_with_letter)
NameError: name 'is_isbn_13' is not defined
```
The experiment exited with exit code 1.


It seems that there was an error in running the experiment due to the `is_isbn_13` function not being properly available in the execution context. I'm going to ensure the correct imports and define both tests to run them properly.

## Experiment Code

I'll re-structure the test functions to include the relevant import statement again.

```python
from string_utils.validation import is_isbn_13

def test_is_isbn_13_malformed_last_character():
    # Invalid ISBN-13 with a non-numeric character
    malformed_isbn = '978031249858Z'  # Should be invalid due to non-numeric character

    output = is_isbn_13(malformed_isbn)

    print(f"Malformed Last Character ISBN Output: {output}")  # Expecting False
    assert output == False, f"Expected False, got {output}"

def test_is_isbn_13_invalid_with_alpha():
    # Invalid ISBN-13 with a letter
    invalid_isbn_with_letter = '978-0-306-4A615-7'  # Should be invalid

    output = is_isbn_13(invalid_isbn_with_letter)

    print(f"Invalid ISBN with Letter Output: {output}")  # Expecting False
    assert output == False, f"Expected False, got {output}"

# Running both tests
test_is_isbn_13_malformed_last_character()
test_is_isbn_13_invalid_with_alpha()
```

Now I'll execute these corrected test functions to see if the mutant produces different outputs compared to the baseline.


### Experiment Results

### Running Experiment on Baseline
```
Malformed Last Character ISBN Output: False
Invalid ISBN with Letter Output: False
```

### Running Experiment on Mutant
```
Malformed Last Character ISBN Output: False
Invalid ISBN with Letter Output: False
```


## Test Instructions

You have 2 experiments/tests left.

Please continue by writing a test that kills the mutant. Remember: the test should pass when executed on the baseline but fail on the mutant.

## Test


## Test
From the earlier experiments, I have consistently observed behavior where both the baseline and the mutant return the same results for the inputs tested. However, one area not thoroughly explored is the incorrect format inputs that might result in the mutant mishandling due to the altered logic regarding the return value of the `is_isbn_13` function.

To specifically target the mutant's change, I will create a test that involves a malformed but appropriately structured ISBN-13 string that may closely resemble a valid ISBN but has a critical flaw that should lead to differing results. 

I will test with a valid-looking ISBN string that has a non-numeric character at the end, such as `978031249858A`. The baseline should return `False`, but I will hypothesize that the mutant might incorrectly validate this.

```python
from string_utils.validation import is_isbn_13

def test_is_isbn_13_invalid_last_character_mutant_killing():
    """
    This test checks a malformed ISBN-13 due to a non-numeric last character.
    The mutant is expected to incorrectly validate this as True, but the baseline
    should return False.
    """
    malformed_isbn = '978031249858A'  # Invalid ISBN-13 due to non-numeric character

    output = is_isbn_13(malformed_isbn)

    print(f"Malformed ISBN with non-numeric character Output: {output}")  # Expecting False
    assert output == False, f"Expected False, got {output}"
```

This test should effectively expose the mutant if it behaves incorrectly due to the logic change introduced in the mutation. 


### Test Results

### Running Test on Baseline

```
Malformed ISBN with non-numeric character Output: False
```


### Running Test on Mutant

```
Malformed ISBN with non-numeric character Output: False
```


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


Given that the last test did not yield the expected results, it is clear that the mutant and baseline code behave identically for the inputs tested so far.

I will think more critically to devise other test cases specifically targeting potential weaknesses introduced by the mutant logic changes, particularly focusing on edge cases with unusual valid characters that may not be rejected properly by the logic of the mutated code.

## New Hypothesis
I suspect that the mutant might improperly handle edge cases, especially when dealing with ISBN-13 validations with malformed inputs that should still be numeric but have mixed characters. Specifically, I will focus on valid lengths but altered formats.

## New Test
I will test an ISBN-like input that looks valid in terms of structure but contains characters that should clearly be invalid, such as `9780312498580B`, which has an extra character at the end. The baseline should reject this with `False`, but if the mutant fails to validate correctly due to the change in `product % 10 <= 0`, it might mistakenly validate it.

```python
from string_utils.validation import is_isbn_13

def test_is_isbn_13_invalid_with_extra_character():
    """
    This test checks an ISBN-13 that has an extra character at the end.
    The baseline should return False, while the mutant might incorrectly return True.
    """
    invalid_isbn = '9780312498580B'  # Invalid ISBN-13 due to extra character

    output = is_isbn_13(invalid_isbn)

    print(f"Invalid ISBN with Extra Character Output: {output}")  # Expecting False
    assert output == False, f"Expected False, got {output}"
```

This test aims to obtain a clear distinction between the baseline and the mutant, assuming the logic change impacts malformed inputs improperly.


### Test Results

### Running Test on Baseline

```
Invalid ISBN with Extra Character Output: False
```


### Running Test on Mutant

```
Invalid ISBN with Extra Character Output: False
```


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


# Conversation Aborted

The conversation has been aborted due to `too_many_turns`

The LLM reached the allowed number of turns.
