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
Given the change in the mutant code, I hypothesize that the mutant will accept email addresses with a tail part that is exactly 255 characters long, while the baseline will reject any email addresses that exceed 255 characters for the tail part. I will test this hypothesis by using an email with a tail of length 255 characters to see if the behavior differs between the baseline and the mutant.

## Experiment
Here is the test case that will check an email input with exactly 255 characters in the tail part after the '@' symbol. I predict that the baseline will return `False`, whereas the mutant will return `True`.

```python
from string_utils.validation import is_email

def test__is_email():
    """
    Test whether an email address with exactly 255 characters in the tail part is accepted.
    The input represents a valid email format with a tail that is exactly 255 characters long.
    I expect the baseline code to return False while the mutant will return True.
    """
    valid_head = 'a' * 64
    valid_tail = 'b' * 255
    email = f"{valid_head}@{valid_tail}"
    
    output = is_email(email)
    print(f"output: {output}")
```


### Experiment Results

### Running Experiment on Baseline
```
output: False
```

### Running Experiment on Mutant
```
output: False
```


## Conclusion
The results from the experiment indicate that both the baseline and the mutant returned `False` for the test case with the email address of length 255 in the tail part. This suggests that the mutant did not change the behavior of the function for this specific test case.

1. The baseline returned `False`, indicating it did not accept the email with a tail of 255 characters.
2. The mutant also returned `False`, suggesting that the alteration in the conditional check did not impact this particular case.
3. More experimentation is needed to find an input that could reveal a difference between the behaviors of the baseline and the mutant.

## Hypothesis
Given the previous results that neither version accepted the email with a tail length of 255, I hypothesize that the mutant may allow more lenient checks in other specific cases. I will check an email address that has a tail part of 254 characters, as this may reveal a difference. Since both versions are supposed to allow at most, exactly, 255 characters but differ on the >= operation, I will test an email with a head of 64 characters and tail of 254.

## Experiment
Here is the test case that will check an email input with 254 characters in the tail. I predict that the baseline will return `True`, whereas the mutant will return `False`.

```python
from string_utils.validation import is_email

def test__is_email_tail_254():
    """
    Test whether an email address with 254 characters in the tail part is accepted.
    The input represents a valid email format with a tail that is 254 characters long.
    I expect the baseline code to return True while the mutant will return False.
    """
    valid_head = 'a' * 64
    valid_tail = 'b' * 254
    email = f"{valid_head}@{valid_tail}"
    
    output = is_email(email)
    print(f"output: {output}")
```


### Experiment Results

### Running Experiment on Baseline
```
output: False
```

### Running Experiment on Mutant
```
output: False
```


## Conclusion
The results from the experiment where I tested an email with a tail part of 254 characters yielded the same result from both the baseline and the mutant, both returning `False`. This indicates that neither version accepted the provided email address.

1. The baseline returned `False`, indicating it rejected the email address.
2. The mutant also returned `False`, suggesting no differing behavior in this case.
3. This experiment suggests a more fundamental issue with the email validation function's logic unrelated to the specific line being altered in the mutant. It may be necessary to explore other parts of the email validation process or different scenarios.

## Hypothesis
Since both the last experiments did not yield differing results, I will hypothesize that the mutant code may allow emails with multiple consecutive dots (')..') in the head part under certain conditions, while the baseline does not. I will construct a test case with two consecutive dots in the head to test this behavior.

## Experiment
Here is the test case that will check an email input with consecutive dots in the head part. I predict that the baseline will return `False` while the mutant will return `True`.

```python
from string_utils.validation import is_email

def test__is_email_consecutive_dots():
    """
    Test whether an email address with consecutive dots in the head part is accepted.
    The input represents a valid email format with two consecutive dots in the head.
    I expect the baseline code to return False while the mutant will return True.
    """
    invalid_head = 'foo..bar'
    valid_tail = 'example.com'
    email = f"{invalid_head}@{valid_tail}"
    
    output = is_email(email)
    print(f"output: {output}")
```


### Experiment Results

### Running Experiment on Baseline
```
output: False
```

### Running Experiment on Mutant
```
output: False
```


## Conclusion
The experiment that involved checking for an email address with consecutive dots in the head part also yielded the same result from both the baseline and the mutant, with both returning `False`.

1. The baseline returned `False`, indicating it rejected the email address with consecutive dots.
2. The mutant also returned `False`, suggesting it did not allow this case either.
3. This indicates that further tests are required to differentiate between the baseline and the mutant since previous inputs have not been successful.

## Equivalent Mutant
At this point, it seems the mutant might be equivalent to the baseline, as all tests performed have shown no differences in their outputs despite testing various edge cases. Both versions reject valid and invalid emails under the same conditions. Further hypothesis formations and experiments are likely to lead to similar results unless a fundamentally different condition is tested. 

However, I will continue to probe the behavior of the mutant versus the baseline until an input is discovered that results in different behavior.

## Hypothesis
I will try testing an email address that has just the maximum length at the boundary defined by `len(tail) = 256`, which is slightly outside the allowed limit for a tail that should be rejected by both versions, since seeds like that are often challenging. 

## Experiment
Here is the test case that will create an email input with a tail part of 256 characters. I predict that both versions will return `False` under these conditions, but I expect the mutant might handle this edge case slightly differently.

```python
from string_utils.validation import is_email

def test__is_email_tail_256():
    """
    Test whether an email address with 256 characters in the tail part is accepted.
    The input represents a valid email format that exceeds the allowed length.
    I expect both baseline and mutant code to return False.
    """
    valid_head = 'a' * 64
    invalid_tail = 'b' * 256
    email = f"{valid_head}@{invalid_tail}"
    
    output = is_email(email)
    print(f"output: {output}")
```


### Experiment Results

### Running Experiment on Baseline
```
output: False
```

### Running Experiment on Mutant
```
output: False
```


## Conclusion
The latest experiment with the email address that has a tail part of 256 characters also resulted in the same output from both the baseline and the mutant, with both returning `False`.

1. The baseline returned `False`, indicating it correctly identified the email as invalid.
2. The mutant also returned `False`, agreeing with the baseline's validation.
3. This repeated result indicates that the mutant still behaves equivalently to the baseline for this particular edge case.

Given the continuous failure to identify a difference in behavior, it may be time to consider that the changes in the mutant may not significantly affect any of the tested inputs, leading to the hypothesis that the mutant is equivalent.

## Equivalent Mutant
I believe the mutant is equivalent. The change in the mutant code modifies the condition regarding the length of the tail in the email validation. However, throughout all experiments conducted (including various edge cases such as tests with tail lengths of 254 and 256), both the baseline and the mutant produced identical outputs of `False`. This suggests that although the allowable condition has changed from `> 255` to `>= 255`, the overall structure and behavior of the validation function for the specific test cases did not change. As no differing behavior has been found after numerous diverse tests, I conclude that this mutant does not introduce a fundamentally new behavior that can be detected through the tests performed thus far.


You claimed that the mutant as equivalent.

Now, try other experiments to check if you can prove yourself wrong. Try new approaches or new inputs you haven't tried yet. See if you can maybe find a difference between the **Baseline** and the **Mutant** anyways.


## Hypothesis
To reassess the equivalence of the mutant, I will explore inputs involving the email structure that are close to boundary limits but include variations not yet tested, such as an email with a very short head segment. This may reveal differing behavior because the mutant relies on the length of the head not exceeding a certain limit, while potentially allowing other structural conditions.

## Experiment
I will test an email with a very short head (less than the allowed maximum) but keep the tail valid under boundary conditions. For example, I will create an email like `'a@' + 'b' * 254`, where the head is only 1 character long and the tail has a valid length of 254 characters.

I predict that the baseline may return `True` while the mutant might handle the edge case differently.

```python
from string_utils.validation import is_email

def test__is_email_short_head():
    """
    Test whether an email address with a very short head and valid tail is accepted.
    The input represents a valid email format, with the head being just 1 character long.
    I expect the baseline code to return True while the mutant may return False.
    """
    short_head = 'a'  # Valid single character head
    valid_tail = 'b' * 254  # Valid tail part
    email = f"{short_head}@{valid_tail}"
    
    output = is_email(email)
    print(f"output: {output}")
```

This will help determine if the mutant still holds equivalently in this situation or if it alters the outcome due to the changes made to the validation logic.

### Also I will now include the reverse case where the tail is exactly 255 (just one character under the limit) but made intentionally invalid under other conditions (repeating characters). 

```python
def test__is_email_tail_255_invalid():
    """
    Test whether an email address with an invalid format but tail part of 255 characters is accepted.
    The input represents an email with invalid head, will check if both versions reject it.
    I expect both baseline and mutant code to return False.
    """
    invalid_head = '.' * 64  # Invalid head with only dots
    valid_tail = 'b' * 255  # Valid tail part
    email = f"{invalid_head}@{valid_tail}"

    output = is_email(email)
    print(f"output: {output}")
```


### Experiment Results

### Running Experiment on Baseline
```
Traceback (most recent call last):
  File "test.py", line 13, in <module>
    test__is_email_tail_255_invalid()
  File "test.py", line 11, in test__is_email_tail_255_invalid
    output = is_email(email)
NameError: name 'is_email' is not defined
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
Traceback (most recent call last):
  File "test.py", line 13, in <module>
    test__is_email_tail_255_invalid()
  File "test.py", line 11, in test__is_email_tail_255_invalid
    output = is_email(email)
NameError: name 'is_email' is not defined
```
The experiment exited with exit code 1.


## Conclusion
The experiments I attempted resulted in a `NameError` because I did not include the necessary import statements for the function `is_email`. This error indicates that the code cannot locate the function being imported, preventing the execution of the test cases. Since both the baseline and mutant produced similar errors for the same input, we still have not encountered any difference in their handling of valid or invalid email structures. 

1. The `NameError` occurred in both experiments, indicating that neither version processed the inputs due to a lack of function definition.
2. This further supports the hypothesis that further structured inputs are needed to challenge the boundary conditions.

## Revised Experiment
To resolve the issue and enable successful execution, I will ensure I import the required function correctly. Then, I will directly retest the previous hypotheses with added error handling to confirm the behavior of the original email checks.

Here's the updated test code:

```python
from string_utils.validation import is_email

def test__is_email_short_head():
    """
    Test whether an email address with a very short head and valid tail is accepted.
    The input represents a valid email format, with the head being just 1 character long.
    I expect the baseline code to return True while the mutant may return False.
    """
    short_head = 'a'  # Valid single character head
    valid_tail = 'b' * 254  # Valid tail part
    email = f"{short_head}@{valid_tail}"
    
    output = is_email(email)
    print(f"output: {output}")

def test__is_email_tail_255_invalid():
    """
    Test whether an email address with an invalid format but tail part of 255 characters is accepted.
    The input represents an email with invalid head, will check if both versions reject it.
    I expect both baseline and mutant code to return False.
    """
    invalid_head = '.' * 64  # Invalid head with only dots
    valid_tail = 'b' * 255  # Valid tail part
    email = f"{invalid_head}@{valid_tail}"

    output = is_email(email)
    print(f"output: {output}")

# Now to run both tests
test__is_email_short_head()
test__is_email_tail_255_invalid()
``` 

This will allow evaluation of how both the baseline and mutant handle variations at the edges of the email validation logic’s boundaries programmatically.


### Experiment Results

### Running Experiment on Baseline
```
output: False
output: False
```

### Running Experiment on Mutant
```
output: False
output: False
```


## Conclusion
Both experiments run successfully, but the results indicate that both the baseline and the mutant still return the same outputs of `False` for each test case.

1. The first test with the email that has a very short head returned `False`, indicating that it was not accepted in either version.
2. The second test with the invalid head (consisting of only dots) also returned `False` for both, confirming that neither version allowed emails in that structure.

Despite these various inputs and the changes made to the email validation logic within the mutant, no differing behavior has been observed yet. 

## Revised Hypothesis 
Given the consistent results across multiple tests, it may be beneficial to explore inputs that deviate further from typical email structures, potentially including malformed formats that might exploit the specific logic differences in the mutated code. 

## New Experiment
I will test an unusual email format that still technically meets the character limit but uses illegal characters in the domain part, for example, using a special character `$` at the end of the email, which is illegal in standard email formats.

This potentially reveals how the mutant's behavior can differ in edge case validation.

### Test Case

```python
def test__is_email_special_character():
    """
    Test whether an email address with an illegal special character at the end is accepted.
    The input uses a valid structure but ends with an illegal character '$'.
    I expect both baseline and mutant code to return False, but will see if they differ.
    """
    valid_head = 'validemail'
    invalid_tail = 'example.com$'  # Illegal character at the end.
    email = f"{valid_head}@{invalid_tail}"

    output = is_email(email)
    print(f"output: {output}")

# Run the test
test__is_email_special_character()
```

This test will help to evaluate if there's any divergence in behavior for formats that still meet length requirements but introduce disallowed characters.


### Test Results

### Running Test on Baseline

```
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__is_email_special_character()
  File "test.py", line 11, in test__is_email_special_character
    output = is_email(email)
NameError: name 'is_email' is not defined
```
The test exited with exit code 1.

Your test needs to pass when executed with the baseline. Adjust the part of your test that caused this error.

### Running Test on Mutant

```
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__is_email_special_character()
  File "test.py", line 11, in test__is_email_special_character
    output = is_email(email)
NameError: name 'is_email' is not defined
```
The test exited with exit code 1.


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


## Revised Hypothesis
To identify a difference between the baseline and the mutant, I will formulate a new hypothesis based on the need to test invalid emails. Specifically, I will construct a case where the email format is valid in structure but contains characters or configurations that are explicitly invalid according to the standard email format rules.

## Adjusting Test Cases
The previous test cases yielded `NameErrors` due to not importing the necessary function, which has been a recurring issue. Since both the baseline and mutant artifacts indicate they have likely comparable validation logic in rejecting structurally valid but conceptually incorrect email formats, I'll craft a new guess for the email validation with a unique combination of allowed and disallowed characters.

### Test Case for Valid Structure but Invalid Email Pattern
Given the mutant's adjustment might allow formats that the baseline rejects, I will create an email with multiple consecutive special characters in the domain part, such as `'valid@ex..ample.com'`.

### Implementing the Test

```python
from string_utils.validation import is_email

def test__is_email_consecutive_special_characters():
    """
    Test whether an email address with consecutive dots in the domain part is accepted.
    The input has a valid structure but contains disallowed consecutive characters.
    I expect the baseline code to return False while the mutant may return True.
    """
    valid_head = 'valid'
    invalid_tail = 'ex..ample.com'  # Invalid due to consecutive dots.
    email = f"{valid_head}@{invalid_tail}"
    
    output = is_email(email)
    print(f"output: {output}")

# Run the test
test__is_email_consecutive_special_characters()
```

### Goal of the Test
1. This test aims to find an inconsistency between the mutant's handling of edge cases containing special character patterns and the baseline's expected rejection of such configurations.
2. I expect the baseline to return `False` but hope to observe an unexpected `True` from the mutant. 

I will run this new test to determine whether it can effectively distinguish between the behaviors of the baseline and the mutant.


### Test Results

### Running Test on Baseline

```
output: False
```


### Running Test on Mutant

```
output: False
```

Your test contains no assertions! Add assertions to your test, so that the test fails when executed with the mutant.

Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


## Test Instructions

You have 2 experiments/tests left.

Please continue by writing a test that kills the mutant. Remember: the test should pass when executed on the baseline but fail on the mutant.

## Test


## Test
To effectively kill the mutant, I will craft a test case that targets the alteration made in the mutant, specifically regarding its handling of the length check for the tail in email validation. 

As previously explored, the change in the mutant allows for accepting a tail of length `>= 255`, whereas the baseline expects `> 255`. The mutant's logic permits the acceptance of an email address that has exactly 255 characters in its tail, and therefore, I can construct a test case specifically for that case.

### Test Case
I will create an email composed of a valid head and a tail that is exactly 255 characters long. This construction should result in the baseline returning `False` while the mutant returns `True`, exploiting the specific condition modified in the code.

```python
from string_utils.validation import is_email

def test__is_email_tail_length_255():
    """
    Test whether an email address with exactly 255 characters in the tail part is accepted.
    The input represents a valid email format with a tail that is exactly 255 characters long.
    I expect the baseline code to return False while the mutant will return True.
    """
    valid_head = 'a' * 64  # 64 characters
    valid_tail = 'b' * 255  # 255 characters
    email = f"{valid_head}@{valid_tail}"

    output = is_email(email)
    assert output == True  # Expecting mutant to accept it and Baseline to reject it
    print(f"output: {output}")

# Run the test
test__is_email_tail_length_255()
```

### Test Goals
- **Baseline**: Should return `False`, indicating it correctly identifies the address as invalid due to the length constraint.
- **Mutant**: Should return `True`, accepting the address due to the relaxed condition in the email validation logic.

Running this test should help define the differing behavior between the two versions, effectively "killing" the mutant if it behaves as expected.


### Test Results

### Running Test on Baseline

```
Traceback (most recent call last):
  File "test.py", line 18, in <module>
    test__is_email_tail_length_255()
  File "test.py", line 14, in test__is_email_tail_length_255
    assert output == True  # Expecting mutant to accept it and Baseline to reject it
AssertionError
```
The test exited with exit code 1.

Your test needs to pass when executed with the baseline. Adjust the part of your test that caused this error.

### Running Test on Mutant

```
Traceback (most recent call last):
  File "test.py", line 18, in <module>
    test__is_email_tail_length_255()
  File "test.py", line 14, in test__is_email_tail_length_255
    assert output == True  # Expecting mutant to accept it and Baseline to reject it
AssertionError
```
The test exited with exit code 1.


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


## Revised Test
It seems that the previous assumption about the output of the baseline may not be correct. The baseline should not necessarily return `False` for a valid email structure, even if the tail's length matches the boundary of 255 characters. Instead, I will modify the test to directly check the behavior of both versions based on the mutation of the length check (`>=` for the mutant vs. `>` for the baseline).

### New Test Case
To ensure the mutant is identified, I will assert that the mutant allows an email with a tail of 255 characters, assuming it should have made adjustments to lengths handling, while the baseline will reject it due to its stricter checks on tail length.

Here’s the corrected approach:

```python
from string_utils.validation import is_email

def test__is_email_tail_length_255():
    """
    Test whether an email address with exactly 255 characters in the tail part is accepted.
    The input represents a valid email format with a tail that is exactly 255 characters long.
    I expect the baseline code to return False while the mutant will return True.
    """
    valid_head = 'a' * 64  # Head is 64 characters
    valid_tail = 'b' * 255  # Tail is 255 characters
    email = f"{valid_head}@{valid_tail}"

    output = is_email(email)
    assert output == False  # Expecting baseline to reject it
    print(f"output: {output}")
    
# Run the test
test__is_email_tail_length_255()
```

### Adjusting the Claims
1. **Expect the Baseline**: To return `False` as it enforces stricter rules on the length of tail characters.
2. **Expect the Mutant**: Should return `True` under the relaxed constraints, accepting it.

### Expectations
If executed correctly:
- Both the baseline should return `False` (successful assertion).
- The mutant should return `True`, resulting in an assertion failure.

With this understanding, I can run this test again to verify the expected differences.


### Test Results

### Running Test on Baseline

```
output: False
```


### Running Test on Mutant

```
output: False
```


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


# Conversation Aborted

The conversation has been aborted due to `too_many_turns`

The LLM reached the allowed number of turns.
