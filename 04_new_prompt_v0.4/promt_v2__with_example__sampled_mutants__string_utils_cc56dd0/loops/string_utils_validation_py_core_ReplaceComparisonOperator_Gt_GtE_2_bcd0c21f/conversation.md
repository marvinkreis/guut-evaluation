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
index 9124cbb..afc6af2 100644
--- a/string_utils/validation.py
+++ b/mutant/string_utils/validation.py
@@ -223,11 +223,11 @@ def is_email(input_string: Any) -> bool:
     try:
         # we expect 2 tokens, one before "@" and one after, otherwise we have an exception and the email is not valid
         head, tail = input_string.split('@')
 
         # head's size must be <= 64, tail <= 255, head must not start with a dot or contain multiple consecutive dots
-        if len(head) > 64 or len(tail) > 255 or head.endswith('.') or ('..' in head):
+        if len(head) > 64 or len(tail) >= 255 or head.endswith('.') or ('..' in head):
             return False
 
         # removes escaped spaces, so that later on the test regex will accept the string
         head = head.replace('\\ ', '')
         if head.startswith('"') and head.endswith('"'):
```

# Debugging


## Hypothesis
I hypothesize that the mutant changes the validation of the tail part of the email address in the `is_email` function. Specifically, the mutant changes the condition from checking if the tail length is less than or equal to 255 to checking if it is greater than or equal to 255. I predict that this will cause the mutant to incorrectly accept email addresses with tail parts that are 255 characters long or greater, while the baseline correctly rejects them. I will verify this by testing an email that has a valid head and a tail exactly 255 characters long.

## Experiment
I will call `is_email` with a valid email structure but with a tail that is exactly 255 characters long, which should be rejected by the baseline but accepted by the mutant.

```python
from string_utils.validation import is_email

def test_is_email():
    # A valid email with a head and a tail of exactly 255 characters.
    head = "user@example.com"
    tail = "a" * 255  # Create a tail with 255 'a' characters
    email = f"{head}@{tail}"
    
    output = is_email(email)
    print(f"Output: {output}")
    assert output == False, f"Expected False, got {output}"
```


### Experiment Results

### Running Experiment on Baseline
```
Output: False
```

### Running Experiment on Mutant
```
Output: False
```


## Conclusion
The experiment shows that both the baseline and the mutant returned `False` when testing an email string with a tail of exactly 255 characters. This contradicts my prediction that the mutant would incorrectly accept this email format due to its modified validation check.

Key conclusions:
- Both the baseline and mutant correctly reject the email with a tail that is exactly 255 characters long. As such, the mutant did not exhibit a different behavior from the baseline based on this input.

## Hypothesis
The previous experiment did not yield a difference between the baseline and the mutant. I will now test an email with a tail that is **greater than** 255 characters. This will validate my hypothesis about the mutant's behavior when the tail has more than 255 characters. I hypothesize that while the baseline should return `False`, the mutant might return `True` due to the change in the condition.

## Experiment
I will call `is_email` with a valid email structure but with a tail that is 256 characters long.

```python
from string_utils.validation import is_email

def test_is_email_long_tail():
    # A valid email with a head and a tail of 256 characters.
    head = "user@example.com"
    tail = "a" * 256  # Create a tail with 256 'a' characters
    email = f"{head}@{tail}"
    
    output = is_email(email)
    print(f"Output: {output}")
    assert output == False, f"Expected False, got {output}"
```


### Experiment Results

### Running Experiment on Baseline
```
Output: False
```

### Running Experiment on Mutant
```
Output: False
```


## Conclusion
The experiment shows that both the baseline and the mutant returned `False` when testing an email string with a tail that is 256 characters long. This means that the mutant did not exhibit different behavior from the baseline for this input either.

Key conclusions:
- Both baseline and mutant correctly reject the email with a tail that is 256 characters long.

## Hypothesis
Since the previous experiments with tails of both 255 and 256 characters did not yield different behaviors for the baseline and the mutant, I will now check inputs that leverage other aspects of the email validation process, specifically the situation where the input string is very close to being valid but contains subtle errors or unexpected characters. 

I hypothesize that the mutant might handle certain malformed email structures differently, particularly focusing on the head part of the email. I will test with an email that has a valid structure but includes multiple consecutive dots in the head section, which should be rejected by both the baseline and the mutant.

## Experiment
Let's create an email with consecutive dots in the head section that should result in rejection. 

```python
from string_utils.validation import is_email

def test_is_email_consecutive_dots():
    # A valid email structure but with consecutive dots in the head
    email = "user..name@example.com"
    
    output = is_email(email)
    print(f"Output: {output}")
    assert output == False, f"Expected False, got {output}"
```


### Experiment Results

### Running Experiment on Baseline
```
Output: False
```

### Running Experiment on Mutant
```
Output: False
```


## Conclusion
The experiment shows that both the baseline and the mutant returned `False` when testing an email string with consecutive dots in the head section. This means that the mutant did not exhibit a different behavior from the baseline once again.

Key conclusions:
- Both baseline and mutant correctly reject the email with consecutive dots in the head section.

## Hypothesis
Since the previous experiments have not yielded different behaviors between the baseline and the mutant, I will change my approach to examine edge cases related to tail length and format. 

Given the changes made in the mutant regarding the tail length condition, I will now test with an email that has a tail length just below the threshold of invalidity by being exactly 256 characters while still ensuring it has a valid structure (no invalid characters). This will test if the mutant still correctly validates the proper structure against the new `>=` condition.

I hypothesize that both the baseline and the mutant should return `True` for this email, but I'll also include invalid cases for comparison.

## Experiment
I will create an email with a head that is valid, but with a tail that is 256 characters long. Then I will check with an invalid head and valid tail.

```python
from string_utils.validation import is_email

def test_is_email_boundary_cases():
    # Valid email with tail of 256 characters
    head = "user@example.com"
    tail = "a" * 256  # 256 characters long
    
    valid_email = f"{head}@{tail}"
    output_valid = is_email(valid_email)
    print(f"Valid email output: {output_valid}")
    
    # Invalid email with tail greater than the maximum length
    head_invalid = "invalid_email@.com"  # Invalid head
    output_invalid = is_email(head_invalid)
    print(f"Invalid email output: {output_invalid}")
    
    # Assertions
    assert output_valid == True, f"Expected True for valid email, got {output_valid}"
    assert output_invalid == False, f"Expected False for invalid email, got {output_invalid}"
```


### Experiment Results

### Running Experiment on Baseline
```
Valid email output: False
Invalid email output: False
Traceback (most recent call last):
  File "test.py", line 20, in <module>
    test_is_email_boundary_cases()
  File "test.py", line 18, in test_is_email_boundary_cases
    assert output_valid == True, f"Expected True for valid email, got {output_valid}"
AssertionError: Expected True for valid email, got False
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
Valid email output: False
Invalid email output: False
Traceback (most recent call last):
  File "test.py", line 20, in <module>
    test_is_email_boundary_cases()
  File "test.py", line 18, in test_is_email_boundary_cases
    assert output_valid == True, f"Expected True for valid email, got {output_valid}"
AssertionError: Expected True for valid email, got False
```
The experiment exited with exit code 1.


## Conclusion
The experiment shows that both the baseline and the mutant returned `False` for the valid email structure with a tail that is 256 characters long, which was unexpected. This indicates that my assumption about what constitutes a valid email format may need to be revisited in relation to the specific length constraints implemented in the email validation function.

Key conclusions:
- Both baseline and mutant incorrectly reject the email with a tail of 256 characters, indicating that the new condition in the mutant did not introduce any acceptance for this case.
- The invalid email with an improper head also correctly returned `False` as expected.


Given this additional insight into how the email validation is functioning, I will now focus on employing an input that is expected to be valid by typical standards but will reveal differences due to the mutant’s altered condition.

## Hypothesis
I will test what happens with a typical valid email format but push the edge conditions by adding a tail that is valid significantly less than 255 characters, perhaps testing with a combination of both valid and clearly invalid characters.

I hypothesize that both the baseline and mutant should validate a typical email format, but I will structure it such that the mutant's alterations may potentially reveal discrepancies based on allowed lengths.

## Experiment
I will create a valid email structure but with a tail of maximum allowable characters while still being valid and with a combination of invalid characters.

```python
from string_utils.validation import is_email

def test_is_email_valid_case():
    # Valid email with a reasonable size for the tail
    head = "user@example.com"  
    tail = "a" * 63  # Maximum allowed length before hitting tail's 64 limit
    valid_email = f"{head}@{tail}"
    
    output_valid = is_email(valid_email)
    print(f"Valid email output: {output_valid}")

    # Now testing edge case with invalid head including invalid characters
    invalid_email = "user.@example.com"  # Invalid due to the dot at the end of the head
    
    output_invalid = is_email(invalid_email)
    print(f"Invalid email output: {output_invalid}")

    # Assertions
    assert output_valid == True, f"Expected True for valid email, got {output_valid}"
    assert output_invalid == False, f"Expected False for invalid email, got {output_invalid}"
```


### Experiment Results

### Running Experiment on Baseline
```
Valid email output: False
Invalid email output: False
Traceback (most recent call last):
  File "test.py", line 21, in <module>
    test_is_email_valid_case()
  File "test.py", line 19, in test_is_email_valid_case
    assert output_valid == True, f"Expected True for valid email, got {output_valid}"
AssertionError: Expected True for valid email, got False
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
Valid email output: False
Invalid email output: False
Traceback (most recent call last):
  File "test.py", line 21, in <module>
    test_is_email_valid_case()
  File "test.py", line 19, in test_is_email_valid_case
    assert output_valid == True, f"Expected True for valid email, got {output_valid}"
AssertionError: Expected True for valid email, got False
```
The experiment exited with exit code 1.


## Conclusion
The results of the experiment show that both the baseline and the mutant returned `False` for the valid email structure with a valid tail. Therefore, my assumption about what constitutes a valid email format is not behaving as expected. This indicates that:

- The email construction utilized in the experiment, despite being syntactically valid, did not meet the functional criteria set by the `is_email` function as defined.
- Both baseline and mutant reject similarly formatted emails, which gives us no indication of the mutant's behavior.

## Hypothesis
Since prior tests have not revealed any different behavior between the baseline and the mutant, I will go back to testing simple, well-defined cases that rely on the edge of valid length but structured correctly. Specifically, I will test a valid email with a structure that is valid under typical circumstances, such as including an atypical character before the '@' but ensuring it fits the maximum expected character count.

I hypothesize that, given certain character constraints, the mutant may not behave as expected due to its modified conditions.

## Experiment
I will construct a generally accepted valid email format, ensuring to fit established conventions and highlight the condition difference in length management due to the adjusted conditions.

```python
from string_utils.validation import is_email

def test_is_email_with_special_characters():
    # Constructing a valid email with special character in the head
    valid_email = "user.name+tag@example.com"  # Valid email
    output_valid = is_email(valid_email)
    print(f"Valid email output: {output_valid}")

    # Also test the expected response for erroneous email structure
    invalid_email = "user..name@example.com"  # Invalid due to consecutive dots
    output_invalid = is_email(invalid_email)
    print(f"Invalid email output: {output_invalid}")

    # Assertions
    assert output_valid == True, f"Expected True for valid email, got {output_valid}"
    assert output_invalid == False, f"Expected False for invalid email, got {output_invalid}"
```


### Experiment Results

### Running Experiment on Baseline
```
Valid email output: True
Invalid email output: False
```

### Running Experiment on Mutant
```
Valid email output: True
Invalid email output: False
```


## Conclusion
The results of the experiment show that both the baseline and the mutant returned `True` for the valid email format and `False` for the email format that had consecutive dots. This once again indicates that there is no change in behavior between the baseline and the mutant for these test cases.

Key conclusions:
- The email `user.name+tag@example.com` is correctly identified as valid by both implementations.
- The email `user..name@example.com` is correctly identified as invalid by both implementations.

## Hypothesis
Since numerous test cases have not revealed any difference in behavior between the baseline and the mutant, I will now focus on testing an email format that generally adheres to typical rules but includes subtle errors that can expose potential flaws in the mutant. 

I hypothesize that if we test an email with an invalid head format where it begins and ends with a special character (e.g., a period), the mutant may handle this differently.

## Experiment
I'll test an email with a valid structure but starts and ends with a special character which is typically not allowed.

```python
from string_utils.validation import is_email

def test_is_email_boundary_special():
    # Constructing an invalid email that starts and ends with special characters
    invalid_email = ".user@example.com."  # Invalid due to being improperly formatted
    output_invalid = is_email(invalid_email)
    print(f"Invalid email output: {output_invalid}")

    # Also test a valid case close to the edge of validity
    valid_email = "user@example.com"
    
    output_valid = is_email(valid_email)
    print(f"Valid email output: {output_valid}")

    # Assertions
    assert output_invalid == False, f"Expected False for invalid email, got {output_invalid}"
    assert output_valid == True, f"Expected True for valid email, got {output_valid}"
```


### Experiment Results

### Running Experiment on Baseline
```
Invalid email output: False
Valid email output: True
```

### Running Experiment on Mutant
```
Invalid email output: False
Valid email output: True
```


## Conclusion
The results of the experiment show that both the baseline and the mutant returned `False` for the invalid email format (".user@example.com.") and `True` for the valid email format ("user@example.com"). This indicates that there is still no behavior difference between the baseline and the mutant for these test cases.

Key conclusions:
- Both implementations correctly identified the email ".user@example.com." as invalid.
- Both implementations correctly identified the email "user@example.com" as valid.

## Hypothesis
Since all tests have thus far demonstrated equivalent behavior between the baseline and the mutant, I will explore inputs that are specifically tailored to their respective boundaries. This time, I will focus on email formats that are designed to be valid according to one set of rules but could be problematic under others.

I hypothesize that testing with an email containing a component that is typically constrained in formats (e.g., a domain that exceeds the usual length conventions or invalid symbols) may reveal discrepancies between the two.

## Experiment
I will test an email structure that violates conventional domain length rules (longer than 253 characters in the domain part).

```python
from string_utils.validation import is_email

def test_is_email_long_domain():
    # Domain portion of valid email construct
    head = "user@example.com"
    long_tail = "a" * 256  # Exceeding maximum domain length
    long_email = f"{head}@{long_tail}"
    
    output_long_email = is_email(long_email)
    print(f"Long domain email output: {output_long_email}")

    # Structuring a valid email for clear comparison
    valid_email = "user@example.com"
    output_valid_email = is_email(valid_email)
    print(f"Valid email output: {output_valid_email}")

    # Assertions
    assert output_long_email == False, f"Expected False for long domain email, got {output_long_email}"
    assert output_valid_email == True, f"Expected True for valid email, got {output_valid_email}"
```


### Experiment Results

### Running Experiment on Baseline
```
Long domain email output: False
Valid email output: True
```

### Running Experiment on Mutant
```
Long domain email output: False
Valid email output: True
```


## Test Instructions

You have 2 experiments/tests left.

Please continue by writing a test that kills the mutant. Remember: the test should pass when executed on the baseline but fail on the mutant.

## Test


## Test
I will create a test case that is designed to reveal the behavior of the mutant specifically based on the condition that was altered within the `is_email` function. The mutant uses `len(tail) >= 255`, which may behave incorrectly under valid constraints. 

To target this, I will use an email with a head that is valid while pushing the limits of the tail's allowable length, specifically designing it as 255 characters long which will be valid under typical rules but could reveal limitations in the mutant handling.

I hypothesize that:
- The baseline should return `True` for this email.
- The mutant will incorrectly return `False` due to the altered length condition.

```python
from string_utils.validation import is_email

def test_is_email_mutant_killing():
    """
    Test the is_email function with a valid email structure
    with a tail of exactly 255 characters. The mutant is expected 
    to fail this test due to its altered condition allowing tails 
    of length >= 255 to be incorrectly processed.
    """
    head = "user@example.com"
    tail = "a" * 255  # Valid tail with maximum length

    email = f"{head}@{tail}"

    output = is_email(email)
    print(f"Output for 255 character tail email: {output}")
    
    assert output == True, f"Expected True for valid email, got {output}"
```


### Test Results

### Running Test on Baseline

```
Output for 255 character tail email: False
Traceback (most recent call last):
  File "test.py", line 19, in <module>
    test_is_email_mutant_killing()
  File "test.py", line 18, in test_is_email_mutant_killing
    assert output == True, f"Expected True for valid email, got {output}"
AssertionError: Expected True for valid email, got False
```
The test exited with exit code 1.

Your test needs to pass when executed with the baseline. Adjust the part of your test that caused this error.

### Running Test on Mutant

```
Output for 255 character tail email: False
Traceback (most recent call last):
  File "test.py", line 19, in <module>
    test_is_email_mutant_killing()
  File "test.py", line 18, in test_is_email_mutant_killing
    assert output == True, f"Expected True for valid email, got {output}"
AssertionError: Expected True for valid email, got False
```
The test exited with exit code 1.


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


## Adjusted Test
Since the previous test resulted in `False` for both the baseline and mutant when it should have passed in the baseline, it's clear that the assumption about the tail’s length being acceptable needs to be revisited based on typical email standards.

The mutant allowed a change to the tail length condition that may result in an unexpected response for emails formatted correctly according to other conventions involving the length of the tail as part of the mutations to the `is_email` function.

## New Hypothesis
Instead of testing for the maximum allowable tail structure, I will create a test case that uses an email with a valid format but whose tail exceeds the maximum traditional length of a domain part when it has acceptable characters, potentially revealing the mutant's failure.

I hypothesize that with an email structure resulting in a tail longer than 255 characters, the baseline should return `False`, while the mutant may incorrectly return `True`.

## Experiment
This experiment will construct an email address where the head is valid and the tail exceeds valid domain character limits.

```python
from string_utils.validation import is_email

def test_is_email_mutant_killing():
    """
    Test the is_email function using an email structure that has a tail
    exceeding valid character limits for email domains (over 255 characters).
    The mutant is expected to incorrectly accept this email due to the altered condition
    on tail length, while the baseline should reject it.
    """
    head = "user@example.com"
    long_tail = "a" * 256  # Exceeding the maximum allowable domain size
    
    email = f"{head}@{long_tail}"

    output = is_email(email)
    print(f"Output for long tail email: {output}")

    # This test is expected to fail on the mutant
    assert output == False, f"Expected False for invalid email, got {output}"
```

This test will clarify the differences in behavior driven by our mutant modifications.


### Experiment Results

### Running Experiment on Baseline
```
Output for long tail email: False
```

### Running Experiment on Mutant
```
Output for long tail email: False
```


# Conversation Aborted

The conversation has been aborted due to `too_many_turns`

The LLM reached the allowed number of turns.
