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
index 9124cbb..ae0093a 100644
--- a/string_utils/validation.py
+++ b/mutant/string_utils/validation.py
@@ -68,11 +68,11 @@ class __ISBNChecker:
 
             try:
                 for index, digit in enumerate(self.input_string):
                     product += int(digit) * (index + 1)
 
-                return product % 11 == 0
+                return product & 11 == 0
 
             except ValueError:
                 pass
 
         return False
```

# Debugging


## Hypothesis
Given the observation that the mutant substitutes the modulo operation `%` with a bitwise AND operation `&`, I hypothesize that the mutation will cause the `is_isbn_10` method to incorrectly validate ISBN-10 numbers. Specifically, I predict that for valid ISBN-10 inputs, the method will return `False`, while for invalid ones, it will return `True`. I will verify this using known valid and invalid ISBN-10 strings.

## Experiment
I will create a test case to check both a valid and an invalid ISBN-10 number: 
1. `is_isbn_10('1506715214')` should return `True` under baseline but will likely return `False` under the mutant.
2. `is_isbn_10('150-6715214')` should also return `True` under baseline but will likely return `False` under the mutant.
3. `is_isbn_10('1234567890')` should return `False` under both the baseline and mutant. 

This will help confirm that the mutant changes the function's intended behavior.

```python
from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    """
    Test the behavior of is_isbn_10 to check the impact of the mutation.
    The valid ISBN '1506715214' should return True, whereas the invalid ISBN '1234567890' should return False.
    We expect the mutant to return False for the valid ISBN and False for the invalid ISBN.
    """
    valid_isbn = '1506715214'
    invalid_isbn = '1234567890'
    
    valid_output = is_isbn_10(valid_isbn)
    invalid_output = is_isbn_10(invalid_isbn)

    print(f"valid_output: {valid_output}")
    print(f"invalid_output: {invalid_output}")

    assert valid_output == True
    assert invalid_output == False
```


### Experiment Results

### Running Experiment on Baseline
```
valid_output: True
invalid_output: False
```

### Running Experiment on Mutant
```
valid_output: True
invalid_output: False
```


## Conclusion
The experiment results show that both the baseline and the mutant returned the same outputs for the tested inputs (`'1506715214'` and `'1234567890'`). This indicates that the change made by the mutant does not affect the correctness of the `is_isbn_10` function in this specific context. Therefore, the mutant appears to be equivalent to the baseline for these test cases.

Given that this mutation doesn't result in differing behaviors for the tested ISBN strings, I will generate new hypotheses and experiments to find another angle to test for the mutation.

## Hypothesis
The change from the modulo operation `%` to the bitwise AND operation `&` might yield different results when applied to specific numerical values. I hypothesize that testing with an ISBN-10 number that contains digits such that the multiplication results in a value that is not perfectly divisible by 11 will allow me to observe a difference in output between the baseline and the mutant.

I will specifically use the invalid ISBN-10 `'123456789X'`, which should raise a ValueError for the baseline by processing the 'X', but might not do so in the mutant due to the behavior change. 

## Experiment
I will add a test case for the ISBN-10 `'123456789X'` to check how it is handled in both the baseline and the mutant for the `is_isbn_10` method:

```python
from string_utils.validation import is_isbn_10

def test__is_isbn_10_with_invalid_character():
    """
    Test the behavior of is_isbn_10 with an invalid ISBN string containing an invalid character 'X'.
    The input '123456789X' should raise a ValueError in baseline but may not in the mutant.
    """
    invalid_isbn_with_char = '123456789X'
    
    try:
        output = is_isbn_10(invalid_isbn_with_char)
        print(f"output: {output}")
        # Expecting a ValueError so we don't reach this assert.
        assert False  # If we reach this line, the test has failed because we expected an exception.
    except ValueError:
        print("ValueError raised as expected.")
```


### Experiment Results

### Running Experiment on Baseline
```
output: False
Traceback (most recent call last):
  File "test.py", line 17, in <module>
    test__is_isbn_10_with_invalid_character()
  File "test.py", line 14, in test__is_isbn_10_with_invalid_character
    assert False  # If we reach this line, the test has failed because we expected an exception.
AssertionError
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
output: False
Traceback (most recent call last):
  File "test.py", line 17, in <module>
    test__is_isbn_10_with_invalid_character()
  File "test.py", line 14, in test__is_isbn_10_with_invalid_character
    assert False  # If we reach this line, the test has failed because we expected an exception.
AssertionError
```
The experiment exited with exit code 1.


## Conclusion
The results of the experiment show that both the baseline and the mutant returned the same output (`False`) for the input `'123456789X'`. Furthermore, neither version raised a `ValueError`, which means that the mutant did not fail as expected. 

This shows that the mutation did not introduce a significant change in behavior for this particular invalid ISBN input. Therefore, the mutant seems to be equivalent again in this context. 

Since the aim is to create a valid mutant-killing test, I will explore the `__ISBNChecker` class further to check for other potential edge cases where the change could lead to observable differences.

## Hypothesis
Next, I hypothesize that using an ISBN-10 number that results in a valid summation that should ideally lead to a `False` result when evaluated is another possible way to differentiate between the baseline and the mutant. Specifically, I will use `'1234567890'`, which should yield an acceptable product for the original function (where `product % 11 == 0`) but likely will not satisfy the mutant (where it uses `product & 11 == 0`).

## Experiment
I will create a test case for this input in order to verify the expected behavior:

```python
from string_utils.validation import is_isbn_10

def test__is_isbn_10_product_edge_case():
    """
    Test the behavior of is_isbn_10 with the ISBN '1234567890'.
    This input should return False for both baseline and mutant; however, 
    the function should evaluate differently due to the mutant's change. 
    We should see False returned, but for the mutant, we check the counts of valid condition checks.
    """
    isbn = '1234567890'
    
    output = is_isbn_10(isbn)
    print(f"output for '1234567890': {output}")

    assert output == False  # Expecting it to return False on both versions.
```


### Experiment Results

### Running Experiment on Baseline
```
output for '1234567890': False
```

### Running Experiment on Mutant
```
output for '1234567890': False
```


## Conclusion
Both the baseline and the mutant returned `False` for the input `'1234567890'`. This confirms the current hypothesis that the underlying logic of the functions in both versions remains equivalent despite the change from the modulo operation to the bitwise AND operation. This indicates that all tested cases have not been able to differentiate between the baseline and the mutant. 

Given that I have not yet found a distinct input that would reveal the mutant's altered behavior, I will reformulate my approach:

## Hypothesis
I hypothesize that leveraging specific edge cases that influence the sum may help distinguish the mutant's behavior. The change affects how numbers evaluate when the resulting product is influenced by the digit multiplier. Testing edge cases such as maximal valid ISBN values that lead to differing outcomes based on their inherent sums could yield results. 

I will choose a known valid ISBN-10 string `'047116072X'` which evaluates to `False` when calculated normally. This string checks if the changed implementation still behaves correctly.

## Experiment
I will implement a test case for this known ISBN-10 string. 

```python
from string_utils.validation import is_isbn_10

def test__is_isbn_10_maximal_case():
    """
    Test the behavior of is_isbn_10 with the ISBN '047116072X'.
    It is a valid ISBN-10 which should evaluate logically but should let us see if the mutant fails to validate it correctly.
    The output should return True under baseline but may return False under mutant due to the bitwise AND change.
    """
    valid_isbn = '047116072X'
    
    try:
        output = is_isbn_10(valid_isbn)
        print(f"output for '047116072X': {output}")
        
        assert output == True  # Expecting True for valid ISBNs in both baseline and mutant.
    except ValueError:
        print("Invalid ISBN raised ValueError as expected under baseline.")
```


### Experiment Results

### Running Experiment on Baseline
```
output for '047116072X': False
Traceback (most recent call last):
  File "test.py", line 18, in <module>
    test__is_isbn_10_maximal_case()
  File "test.py", line 15, in test__is_isbn_10_maximal_case
    assert output == True  # Expecting True for valid ISBNs in both baseline and mutant.
AssertionError
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
output for '047116072X': False
Traceback (most recent call last):
  File "test.py", line 18, in <module>
    test__is_isbn_10_maximal_case()
  File "test.py", line 15, in test__is_isbn_10_maximal_case
    assert output == True  # Expecting True for valid ISBNs in both baseline and mutant.
AssertionError
```
The experiment exited with exit code 1.


## Conclusion
The results indicate that both the baseline and the mutant returned `False` for the input `'047116072X'`. Since this ISBN-10 is expected to validate as valid (True), failing to do so under both versions suggests that both are returning equivalent behaviors when handling this input. 

This leads to the conclusion that numerous attempts to find a difference between the two versions have not yielded a viable test that demonstrates the mutant's altered behavior. 

## Test
To solidify the investigation, I will now formulate a test based on my findings. Given that I was unable to identify inputs that can effectively differentiate between the baseline and the mutant for the `is_isbn_10` function, I will conclude that the mutant is indeed equivalent.

The following code represents a test which checks for the correctness of a valid input that should logically return valid but ultimately returns False in both versions:

```python
from string_utils.validation import is_isbn_10

def test__is_isbn_10_equivalence():
    """
    Test the behavior of is_isbn_10 on a known valid ISBN-10 number.
    Both the baseline and the mutant are expected to return False for this factorial input.
    This suggests they are behaving equivalently under the same conditions.
    The ISBN '047116072X' should ideally return True but instead returns False for both.
    """
    isbn = '047116072X'  # A valid ISBN-10
    output = is_isbn_10(isbn)
    
    print(f"output for '047116072X': {output}")
    assert output == False  # This demonstrates that both are equivalent as neither can validate it properly.
```


### Test Results

### Running Test on Baseline

```
output for '047116072X': False
```


### Running Test on Mutant

```
output for '047116072X': False
```


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


## Conclusion from Test Results
The test results show that both the baseline and the mutant produced the same output (`False`) for the input `'047116072X'`. This means that the test did not manage to differentiate between the two versions correctly. The mutant has not been killed since it behaves identically to the baseline for this input.

To move forward, I'll need to continue searching for another specific input that can help to reveal the mutant's changed behavior caused by the alteration from modulo `%` to bitwise AND `&`.

## New Hypothesis
I hypothesize that there are specific inputs that, due to the nature of bitwise operations, may pass through the mutant but fail on the baseline. More specifically, I will test a case of an ISBN-10 that evaluates to a product just below a multiple of 11 (like `0` for the `& 11` condition).

To determine this, I will check an ISBN with a calculated product of `11`, such as `isbn = '123456789X'`, expecting it to return `True` on the baseline (valid), whereas the mutant might yield `False` as it checks with the bitwise AND against `11`.

## Experiment
I will implement this hypothesis and create a test for this ISBN-10:
```python
from string_utils.validation import is_isbn_10

def test__is_isbn_10_boundary_case():
    """
    Test the behavior of is_isbn_10 with the ISBN '123456789X'.
    This input checks the boundaries of the operation where modulo would yield a valid result, but 
    the bitwise AND will not, thus differentiating the mutant from the baseline.
    The baseline should return True; the mutant is expected to return False.
    """
    isbn = '123456789X'  # Known ISBN-10 that is valid
    
    output = is_isbn_10(isbn)
    print(f"output for '123456789X': {output}")
    
    assert output == True  # Expecting a valid response from the baseline.
```
This test should help identify differing behavior between the baseline and the mutant based on their divergent handling of the product calculations.


### Experiment Results

### Running Experiment on Baseline
```
output for '123456789X': False
Traceback (most recent call last):
  File "test.py", line 16, in <module>
    test__is_isbn_10_boundary_case()
  File "test.py", line 15, in test__is_isbn_10_boundary_case
    assert output == True  # Expecting a valid response from the baseline.
AssertionError
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
output for '123456789X': False
Traceback (most recent call last):
  File "test.py", line 16, in <module>
    test__is_isbn_10_boundary_case()
  File "test.py", line 15, in test__is_isbn_10_boundary_case
    assert output == True  # Expecting a valid response from the baseline.
AssertionError
```
The experiment exited with exit code 1.


## Conclusion
The results of the most recent experiment showed that both the baseline and the mutant returned `False` for the input `'123456789X'`. This input was expected to be recognized as valid ISBN-10, highlighting that both implementations are still behaving equivalently and not differentiating.

This indicates that, after multiple tests and investigations, we have not successfully uncovered any distinguishing behavior between the baseline and the mutant for the `is_isbn_10` function.

## Equivalent Mutant
I believe the mutant is equivalent. The change from the modulo operation `%` to bitwise AND `&` does not affect the boolean logic outcome in the scope of ISBN validation for the tested inputs. All the ISBN checks that pass through this method do not cause a detectable difference because the overall sum calculations do not yield scenarios where the bitwise operation could differentiate from modulo operation usage effectively.

Despite multiple attempts to identify specific cases that might reveal a differential handling due to the mutant's change, the explored ISBN values do not lead to varying outputs, highlighting that they are effectively equivalent in practical operational outcomes.

### Next Steps
At this point, further exploration may involve direct computational steps to verify the outputs of calculations used in `is_isbn_10`. However, based on the current progress and hypotheses, it seems we have exhausted useful angles of testing with the established conditions. 

If new input cases or structured edge cases arise, I will explore those to definitively assert divergent behavior with future experiments.


You claimed that the mutant as equivalent.

Now, try other experiments to check if you can prove yourself wrong. Try new approaches or new inputs you haven't tried yet. See if you can maybe find a difference between the **Baseline** and the **Mutant** anyways.


## New Experiment Approach
To attempt to find a difference between the baseline and the mutant, I will focus on inputs where the ISBN evaluation causes the product to be around or equal to a threshold value that affects both the modulo calculation and bitwise AND differently. 

Given the transformations, I can create an ISBN number that will yield varying outcomes by tricking the weightings of the digits in such a way that it allows for observable differences.

### New Hypothesis
I hypothesize that testing the function with an ISBN that includes varied digits, particularly `isbn = '1234567890'`, could yield different behavior by forcing an edge case scenario whereby the return of the functions diverges under bitwise adjustments to their logical states.

The ISBN `1234567890` will be evaluated such that the weighted product leads to boundary conditions where the division aspects differ. The mutant version, using `product & 11`, may not pass as the bitwise outcome does not evaluate similarly as modulo.

### New Experiment
Here's how I will structure this new test:

```python
from string_utils.validation import is_isbn_10

def test__is_isbn_10_weight_condition():
    """
    Test the behavior of is_isbn_10 with the ISBN '1234567890'.
    This test uses a simple increasing sequence that under normal conditions should yield 
    a true return via the base checks but should flag differently under the mutant's 
    alteration due to the bitwise processes.
    We expect the baseline to return True, while the mutant may return False.
    """
    isbn = '1234567890'  # This is commonly seen as valid
    
    output = is_isbn_10(isbn)
    print(f"output for '1234567890': {output}")

    assert output == True  # Expecting a valid response in the baseline.
```

This test aims to verify if any outcome differences materialize through checks on both versions when validating this specific input ISBN-10 number. The goal is to reveal any divergence through the applied logic changes.


### Experiment Results

### Running Experiment on Baseline
```
output for '1234567890': False
Traceback (most recent call last):
  File "test.py", line 17, in <module>
    test__is_isbn_10_weight_condition()
  File "test.py", line 16, in test__is_isbn_10_weight_condition
    assert output == True  # Expecting a valid response in the baseline.
AssertionError
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
output for '1234567890': False
Traceback (most recent call last):
  File "test.py", line 17, in <module>
    test__is_isbn_10_weight_condition()
  File "test.py", line 16, in test__is_isbn_10_weight_condition
    assert output == True  # Expecting a valid response in the baseline.
AssertionError
```
The experiment exited with exit code 1.


## Conclusion
The experiment shows that both the baseline and the mutant returned `False` for the ISBN `1234567890`. This result is not surprising, as this ISBN-10 is indeed not valid based on the established checksum calculations.

### Ongoing Observations
Having run multiple tests, including the ISBN values that should logically function correctly, it appears so far that both the baseline and the mutant are yielding equivalent results across various valid and invalid ISBN strings. Each test has demonstrated that the modifications from `%` to `&` lead to no discernible difference in their behaviors for the inputs attempted.

## Investigating New Scenarios
Recognizing the paired outputs of `False` across valid ISBN tests, I will now employ a different method: I will create a batch of edge-case ISBN-10 inputs with the expectation of yielding valid results from the baseline and invalid from the mutant. 

### New Hypothesis
I hypothesize that inputs with odd configurations of digits could lead us to observe a difference in outputs, particularly ones where expected sums might not align when subjected to the bitwise AND. For instance, strings that are very close to being valid but not within checksum requirements particularly could reveal a difference.

### New Experiment
I will deploy the ISBN `014300723X` which is known to be valid and check for the workings of both functions.

Here is the new test:

```python
from string_utils.validation import is_isbn_10

def test__is_isbn_10_almost_valid():
    """
    Test the behavior of is_isbn_10 with the input '014300723X'.
    This ISBN-10 is known valid, and we should expect it to return True for the baseline.
    The mutant may fail to recognize the valid ISBN due to the alteration in the logic
    from modulo to bitwise AND. We expect baseline to return True while mutant may return False.
    """
    isbn = '014300723X'  # A known valid ISBN-10

    output = is_isbn_10(isbn)
    print(f"output for '014300723X': {output}")

    assert output == True  # Expecting valid ISBN-10 from the baseline.
```

This test aims to verify if any output differences emerge when subjected to both baseline and mutant evaluations of this specific known valid ISBN-10 number.


### Experiment Results

### Running Experiment on Baseline
```
output for '014300723X': False
Traceback (most recent call last):
  File "test.py", line 16, in <module>
    test__is_isbn_10_almost_valid()
  File "test.py", line 15, in test__is_isbn_10_almost_valid
    assert output == True  # Expecting valid ISBN-10 from the baseline.
AssertionError
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
output for '014300723X': False
Traceback (most recent call last):
  File "test.py", line 16, in <module>
    test__is_isbn_10_almost_valid()
  File "test.py", line 15, in test__is_isbn_10_almost_valid
    assert output == True  # Expecting valid ISBN-10 from the baseline.
AssertionError
```
The experiment exited with exit code 1.


## Test Instructions

You have 2 experiments/tests left.

Please continue by writing a test that kills the mutant. Remember: the test should pass when executed on the baseline but fail on the mutant.

## Test


To formulate a test that can effectively kill the mutant, I will focus on using known ISBN-10 values that are deliberately configured to test the boundary conditions of the alteration from modulo `%` to bitwise AND `&`. 

After reviewing the previous input tests, I plan to create an explicit scenario with a sum of weighted digits that yields a product just fitting for the detection of discrepancies.

### Selected Test Case
I will use the ISBN `0306406152`. This is a valid ISBN-10 under the baseline and is expected to return `True`. However, due to the mutant's modification, it may return `False` because of how the product total interacts with the subsequent validation check.

### Test Code
Here is the test case designed to kill the mutant:

```python
from string_utils.validation import is_isbn_10

def test__kill_mutant_isbn():
    """
    Test the behavior of is_isbn_10 with the valid ISBN '0306406152'.
    The expected behavior is that it should return True for the baseline due to being a valid ISBN-10.
    The mutant is expected to return False due to its altered logic via the AND operation instead of modulo.
    """
    isbn = '0306406152'  # A valid ISBN-10
    
    output = is_isbn_10(isbn)
    print(f"output for '0306406152': {output}")

    assert output == True  # Expecting it to validate successfully in the baseline but likely fail in the mutant.
```

This test checks the ISBN against the baseline and expects it to pass (return `True`) indicating valid, whereas the mutant's logic change should lead to a `False` outcome. The test will confirm if the mutant behaves differently due to its changed operational conditions.


# Results

The LLM found a test case that detects the mutant.

## Mutant

```diff mutant.diff
diff --git a/string_utils/validation.py b/mutant/string_utils/validation.py
index 9124cbb..ae0093a 100644
--- a/string_utils/validation.py
+++ b/mutant/string_utils/validation.py
@@ -68,11 +68,11 @@ class __ISBNChecker:
 
             try:
                 for index, digit in enumerate(self.input_string):
                     product += int(digit) * (index + 1)
 
-                return product % 11 == 0
+                return product & 11 == 0
 
             except ValueError:
                 pass
 
         return False
```

## Test Case

```python
from string_utils.validation import is_isbn_10

def test__kill_mutant_isbn():
    """
    Test the behavior of is_isbn_10 with the valid ISBN '0306406152'.
    The expected behavior is that it should return True for the baseline due to being a valid ISBN-10.
    The mutant is expected to return False due to its altered logic via the AND operation instead of modulo.
    """
    isbn = '0306406152'  # A valid ISBN-10
    
    output = is_isbn_10(isbn)
    print(f"output for '0306406152': {output}")

    assert output == True  # Expecting it to validate successfully in the baseline but likely fail in the mutant.
```

## Running Test on Baseline

```
output for '0306406152': True
```

## Running Test on Mutant

```
output for '0306406152': False
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__kill_mutant_isbn()
  File "test.py", line 14, in test__kill_mutant_isbn
    assert output == True  # Expecting it to validate successfully in the baseline but likely fail in the mutant.
AssertionError
```
The test exited with exit code 1.
