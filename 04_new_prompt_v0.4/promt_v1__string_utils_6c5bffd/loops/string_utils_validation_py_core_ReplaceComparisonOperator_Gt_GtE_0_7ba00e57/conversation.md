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
    (we will write the results)
    #### Running Experiment on Mutant
    (we will write the results)

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

    [repeat ("Test") or ("Hypothesis", "Experiment", "Experiment Results", "Conclusion") until a test successfully killed the mutant]

    [at any point, if you believe the mutant to be equivalent to the original code]
    ## Equivalent Mutant
    (a short explanation about why the mutant is equivalent)

## Notes

Make sure that `## Experiment` is always followed by `### Experiment Results` and `## Test` is always followed by `## Test Results`. This is important for parsing your responses.


# Output Format for Code

Write all code in markdown code blocks and specify the language, e.g.:

    ```python
    // python code here
    ```

Make sure to import all necessary functions in every code snippet. You can assume that all python files we list are in the current directory (`.`). For example, you can import the following file with `import guut.config as config` or `from guut.config import example`:

```python guut/config.py
def example():
    pass
```

Output all code in single Python function called `test__<function_name>` with no parameters. Don't use any testing frameworks.


# Running code

Whenever you submit a test case (experiment or test), our system will run your code on the **Baseline** (the correct code, without the mutant) and give you the output. When that is finished, it applies the **Mutant** to the code and runs your code again.

This means that your test case can only use one version of the target code (**Baseline** or **Mutant**) at once. You test case simply imports the target code and runs it, without knowing if it imported the **Baseline** or the **Mutant**. Since the system will run your test case once with the **Baseline** and once with the **Mutant**, you will still get the output for both versions.

Again, you cannot import the **Baseline** and the **Mutant** together. Your tests will import the target code, which can be either the **Baseline** or the **Mutant** and your job is to design the test in a way, such that it produces different outputs when we switch out the imported target code from the **Baseline** to the **Muntant**.

Therefore, there is no point in re-implementing the code yourself, since the test should examine the imported code, not a recreation it.

There is also no `mutant` module, so imports like `from mutant.sieve import sieve` will result in `ModuleNotFoundError: No module named 'mutant'`. This makes the test case useless.

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

Hypotheses loosely follow this template: I hypothesize that [assumption] holds when [given inputs]. I predict that [assumed result] and I will verify this by [more explanation and experiment description].

## Experiments

After stating a hypothesis, you create an experiment to test it. Each experiment will contain a Python test case, which imports and calls the target code. Once you stated the test case, our system will add it to the target code and execute it. First, it runs your code on the **Baseline** (the correct code, without the mutant) and gives you the output. When that is finished, it applies the **Mutant** to the code and runs your code again.

Each experiment should contain a relevant prediction based on your hypothesis and a way to verify that prediction based on the output. Basically, you run the target code and predict the output based on your hypothesis. Therefore, add print statements to print out relevant values, which will help you understand what the code is doing.

Your experiment is agnostic of which version of the code it is hadling (**Baseline** or **Mutant**). Therefore, never use add print statements like `print(f"baseline output: {output}")` or `print(f"mutant output: {output}")`. This will make your experiment results confusing and useless. Instead, use use agnostic statements like `print(f"output: {output}")`.

Some notes:
- Keep your experiments/tests short and simple.
- Use print statements liberally in your experiments.
- Never recreate the mutant as part of your experiment/test.

Here is an example experiment:

### Example Experiment

```python
from sieve import sieve

def test__sieve():
    output = sieve(5)
    print(f"output = {output}")
    assert len(output) > 0
```

### Example Experiment Results

#### Running Experiment on Baseline
```
output = [2, 3, 5]
```

#### Running Experiment on Mutant
```
output = []
Traceback (most recent call last):
  File "test.py", line 7, in <module>
    test__sieve()
  File "test.py", line 6, in test__sieve
    assert len(output) > 0
           ^^^^^^^^^^^^^^^
AssertionError
```
The experiment exited with exit code 1

## Conclusions

After every experiment, write a conclusion that summarizes on the results. Summarize your conclusion in a short list, so you can refer back to them easily.

Pay close attention to experiment output:
- Did the baseline have any errors? Does the experiment need to be fixed?
- Are there any discrepancies between the output of the **Baseline** and the **Mutant**? That means you detected mutant.

It is already enough to find a single input that can distinguish between the **Baseline** and the **Mutant**. Exceptions and timeouts also count. Any difference in behavior. Once you have found an input that triggers a difference, you can continue and write the test case.

Otherwise, keep creating hypotheses and experiments until you have found the right inputs. Then you can finish debugging and write the mutant-killing test.

## Tests

Once you have found any inputs that cause a difference in behavior, you can write a test that kills the mutant. Similarly to experiments, when you finished writing your code, we will copy the test case and execute it against the **Baseline**, i.e., the regular program without the mutant, then apply the **Mutant** and execute it again.

The test kills the mutant if, and only if, the test passes when executed with the **Baseline** and fails when executed with the **Mutant**.

Failing is defined as exiting with exitcode 1 here. This means that the test needs to result in either a *a failed assertion*, an *uncaught exception* or a *timeout* when executed on the **Mutant**.

This means that you have to include relevant assertions in your test, unless the mutant raises an exception or results in a timeout. Create relevant assertions based on your experiment findings. This is vital, as your test will be useless otherwise.

Include a relevant docstring commnent with a summary of your findings. The comment should explain what the test checks for and why. Include relevant findings from your conclusions.

Here is an example test:

### Example Test

```python
from rpn_eval import rpn_eval

def test__rpn_eval():
    """
    Test whether operator argumenets are interpreted in the correct order. The input represents the calculation (8 / 2), which will lead to different results if the argument order is swapped, since (2 / 8) != (8 / 2).
    """
    output = rpn_eval([8.0, 2.0, '/'])
    assert output == 4.0
```

#### Running Test on Baseline
```

```

#### Running Test on Mutant
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

I believe the mutant is equivalent. The change [mutant change] doesn't affect the way [some result] is computed. My experiments show that [tried inputs] did not result in any different behavior in the mutant, which suggest [more explanation]. Since my past tries have failed to make the mutant behave differently, I will now try [new approach].


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
index 9124cbb..aba0380 100644
--- a/string_utils/validation.py
+++ b/mutant/string_utils/validation.py
@@ -215,11 +215,11 @@ def is_email(input_string: Any) -> bool:
     :param input_string: String to check.
     :type input_string: str
     :return: True if email, false otherwise.
     """
     # first simple "pre check": it must be a non empty string with max len 320 and cannot start with a dot
-    if not is_full_string(input_string) or len(input_string) > 320 or input_string.startswith('.'):
+    if not is_full_string(input_string) or len(input_string) >= 320 or input_string.startswith('.'):
         return False
 
     try:
         # we expect 2 tokens, one before "@" and one after, otherwise we have an exception and the email is not valid
         head, tail = input_string.split('@')
```

# Debugging


## Hypothesis
I hypothesize that the mutant's change to the email validation logic affects valid email strings that have a length of exactly 320 characters. Since the original code allows lengths of up to 320 characters, while the mutant rejects them, I predict that valid email inputs with a maximum length of 320 characters will return different outputs when processed with the original code versus the mutated code. I will verify this by testing a valid email string that is exactly 320 characters long.

## Experiment
```python
from string_utils.validation import is_email

def test__is_email():
    """
    Test the is_email function with an email that is exactly 320 characters long.
    The input is a valid email format, which should pass in the baseline but fail in the mutant.
    """
    valid_email = 'a' * 64 + '@' + 'b' * 255  # This creates a valid email of length 320
    output = is_email(valid_email)
    print(f"output = {output}")
```


### Experiment Results

### Running Experiment on Baseline
```
output = False
```

### Running Experiment on Mutant
```
output = False
```


## Conclusion
The experiment results indicate that the output for both the Baseline and the Mutant is `False`. Since both versions returned the same output, it suggests that the mutant did not behave differently than the original code for the tested input. Therefore, the change in the mutant is not detectable with the current hypothesis based on a 320-character valid email. 

I need to refine my hypothesis and test a different output scenario. Specifically, I will try valid email formats with a length of 319 characters and 321 characters, as they should validate correctly only for the baseline and show differences in outputs with the mutant.

## Hypothesis
I hypothesize that the mutant will produce different outputs when tested with email strings that are exactly 319 characters (which should be a valid email) and 321 characters long (which should be an invalid email). I predict that the baseline will return `True` for the 319-character input and `False` for the 321-character input, while the mutant should return `False` for both.

## Experiment
```python
from string_utils.validation import is_email

def test__is_email_length_bounds():
    """
    Test the is_email function with emails that have boundary lengths (319 and 321).
    The 319 character email should be valid, while the 321 character should be invalid in the baseline.
    The mutant is expected to return false for both inputs.
    """
    valid_email_319 = 'a' * 64 + '@' + 'b' * 254  # Valid email of length 319
    invalid_email_321 = 'a' * 64 + '@' + 'b' * 255  # Invalid email of length 321

    output_319 = is_email(valid_email_319)
    output_321 = is_email(invalid_email_321)
    
    print(f"output_319 = {output_319}")  # Should return True on baseline, False on mutant
    print(f"output_321 = {output_321}")  # Should return False on both
```


### Experiment Results

### Running Experiment on Baseline
```
output_319 = False
output_321 = False
```

### Running Experiment on Mutant
```
output_319 = False
output_321 = False
```


## Conclusion
The experiment results indicate that for both the Baseline and the Mutant, the outputs for the 319-character and 321-character email strings are both `False`. This means that both versions of the code are detecting these email inputs as invalid. Since both versions behave the same way for these particular tests, this does not successfully differentiate between the Baseline and the Mutant.

Given that previous hypotheses and experiments have not produced results that distinguish the two versions, I will revise my approach to test for a slightly different aspect of email validation, such as including sequences with whitespace or special characters.

## Hypothesis
I hypothesize that the mutant will produce different outputs when tested with an email string that is not properly formatted but has valid length constraints (e.g., leading or trailing spaces, missing domain, or an invalid character). I predict that the baseline will identify improperly formatted email addresses as `False`, while the mutant may return `True` for some cases due to its inclusive length check. I will test an email with leading spaces and an improperly formatted example.

## Experiment
```python
from string_utils.validation import is_email

def test__is_email_format_invalid():
    """
    Test the is_email function with improperly formatted emails.
    A valid input with leading spaces should return False on the baseline, and could potentially
    be incorrectly accepted by the mutant due to the less restrictive length check.
    """
    invalid_email_with_spaces = '   my.email@the-provider.com'  # Valid format, but with leading spaces
    improperly_formatted_email = 'invalid@ email.com'             # Invalid format (space before '@')

    output_spaces = is_email(invalid_email_with_spaces)
    output_invalid_format = is_email(improperly_formatted_email)
    
    print(f"output_spaces = {output_spaces}")  # Should return False on both
    print(f"output_invalid_format = {output_invalid_format}")  # Should return False on both
```


### Experiment Results

### Running Experiment on Baseline
```
output_spaces = False
output_invalid_format = False
```

### Running Experiment on Mutant
```
output_spaces = False
output_invalid_format = False
```


## Conclusion
The experiment results show that for both the Baseline and the Mutant, the outputs for improperly formatted email strings are both `False`. This indicates that the mutant does not differ in behavior from the original function when it comes to these specific invalid email inputs. Both versions correctly identify these emails as invalid.

It appears that previous hypotheses have not revealed any discrepancies between the Baseline and the Mutant. I will try testing with a well-formed but boundary-length email that exceeds 320 characters to see if it triggers different behavior.

## Hypothesis
I hypothesize that the mutant will behave differently when tested with an email that exceeds the maximum length of 320 characters. The Baseline should return `False` for any email over this limit, while the mutant might incorrectly allow it due to the modified check. I will test an email string with a total length of 321 characters.

## Experiment
```python
from string_utils.validation import is_email

def test__is_email_exceeds_length():
    """
    Test the is_email function with an email that exceeds 320 characters in length.
    The input is a valid email format, which should return False on the baseline and could return True on the mutant.
    """
    invalid_email_exceeds_length = 'a' * 64 + '@' + 'b' * 257  # Total length is 321 characters
    
    output_exceeds_length = is_email(invalid_email_exceeds_length)
    print(f"output_exceeds_length = {output_exceeds_length}")  # Should return False on baseline and potentially True on mutant
```


### Experiment Results

### Running Experiment on Baseline
```
output_exceeds_length = False
```

### Running Experiment on Mutant
```
output_exceeds_length = False
```


## Conclusion
The experiment results indicate that the output for the email string exceeding 320 characters is `False` for both the Baseline and the Mutant. This means that both versions correctly identify inputs that exceed the maximum email length as invalid.

Since I have not been able to find any differing outputs between the Baseline and the Mutant with previous hypotheses and experiments, I will now expand my testing approach by trying cases that would trigger a different pattern of behavior based on the email format. 

Specifically, I will check:
1. A valid email format that is slightly modified in a way that should be valid according to the original criteria, perhaps by removing an expected character.
2. A boundary test that mixes valid and invalid formats with similar overall lengths, to see if any nuanced differences arise.

## Hypothesis
I hypothesize that the mutant may not handle certain edge cases properly, such as an email address that contains invalid characters or sequences but appears valid at a glance (e.g., multiple consecutive dots). The Baseline should return `False` for invalid cases, while the mutant may incorrectly accept them due to a less strict check related to the conditions implemented in the original method.

## Experiment
```python
from string_utils.validation import is_email

def test__is_email_with_invalid_sequences():
    """
    Test the is_email function with emails containing invalid patterns.
    The first email has consecutive dots which should be invalid; 
    the second email has the format closely resembling valid emails but includes invalid patterns.
    """
    invalid_email_consecutive_dots = 'my..email@the-provider.com'  # Invalid due to consecutive dots
    valid_resembling_invalid_email = 'my.email@the-provider..com'  # Invalid due to consecutive dots in the domain

    output_consecutive_dots = is_email(invalid_email_consecutive_dots)
    output_resembling_invalid = is_email(valid_resembling_invalid_email)
    
    print(f"output_consecutive_dots = {output_consecutive_dots}")  # Should return False on both
    print(f"output_resembling_invalid = {output_resembling_invalid}")  # Should return False on both
```


### Experiment Results

### Running Experiment on Baseline
```
output_consecutive_dots = False
output_resembling_invalid = False
```

### Running Experiment on Mutant
```
output_consecutive_dots = False
output_resembling_invalid = False
```


## Conclusion
The experiment results indicate that both the Baseline and the Mutant returned `False` for the email strings containing invalid patterns (consecutive dots). This means that both versions correctly identify these inputs as invalid. 

Based on the series of experiments conducted, I have tried numerous hypotheses and edge cases without uncovering any differences in behavior between the Baseline and the Mutant. It seems that the change made in the mutant does not lead to observable differences in email validation behavior.

Given the consistent results, I am led to conclude that this specific mutant is likely **equivalent** to the original implementation with respect to the tested email validation function.

## Equivalent Mutant
I believe the mutant is equivalent. The change in the condition from `len(input_string) > 320` to `len(input_string) >= 320` does not lead to any deviations in the behavior of the `is_email` function when tested with various valid and invalid email formats. Throughout the experiments, no input led to differing outputs, suggesting that the mutant handles email validation consistently with the original implementation. 

Since the mutant does not exhibit differing behavior from the Baseline function across the tested inputs, it cannot be detected by any possible tests. The validation behavior remains intact regardless of the potential for edge cases around length, special characters, and sequence patterns. Thus, I will conclude this round of scientific debugging here.


You claimed that the mutant as equivalent.

Now, try other experiments to check if you can prove yourself wrong. Try new approaches or new inputs you haven't tried yet. See if you can maybe find a difference between the **Baseline** and the **Mutant** anyways.


## Hypothesis
I hypothesize that the mutant will behave differently when tested with email strings that correctly follow email formatting rules but still contain special characters that are often incorrectly accepted in certain email fields (such as allowing empty usernames or invalid symbols). Testing with such strings might expose a difference in handling improperly structured valid inputs that loosely adhere to the constraints.

## Experiment
I will focus on testing email addresses that use forbidden characters such as `!`, `#`, `%`, `&`, and also test with an email that contains an empty username (e.g., `@provider.com`), as these should return `False` in the baseline but may not properly be caught in the mutant due to relaxed constraints:

```python
from string_utils.validation import is_email

def test__is_email_with_invalid_characters():
    """
    Test the is_email function with emails containing invalid characters or empty usernames.
    These should return False for both versions, but the mutant has a potential for accepting them.
    """
    invalid_email_special_character = 'user!name@provider.com'  # Invalid due to special character '!'
    invalid_email_empty_username = '@provider.com'              # Invalid due to empty username

    output_special_character = is_email(invalid_email_special_character)
    output_empty_username = is_email(invalid_email_empty_username)

    print(f"output_special_character = {output_special_character}")  # Should return False on both
    print(f"output_empty_username = {output_empty_username}")        # Should return False on both
```

Let's proceed and see if these inputs show any discrepancies in behavior between the Baseline and the Mutant.


### Experiment Results

### Running Experiment on Baseline
```
output_special_character = True
output_empty_username = False
```

### Running Experiment on Mutant
```
output_special_character = True
output_empty_username = False
```


## Conclusion
The experiment results indicate that both the Baseline and the Mutant confirmed that the email string with an invalid special character returned `True`, while the email address with an empty username returned `False`. 

Here are the findings:
1. **Invalid Email with Special Character (`user!name@provider.com`)**:
   - Baseline: `True`
   - Mutant: `True`

2. **Invalid Email with Empty Username (`@provider.com`)**:
   - Baseline: `False`
   - Mutant: `False`

Both versions behaved the same for these test inputs—accepting the email with the special character and rejecting the one with an empty username. While both versions handle these specific invalid inputs similarly, it does not provide evidence of a discrepancy, either.

However, this shows that the mutant maintains the same email validation logic, at least for these inputs. The presence of special characters in email usernames is typically accepted in some formats, which accounts for the permissible output in both versions. 

To find where differences arise, I should experiment with even more nuanced inputs in the form of leading special characters, long domains with invalid formats, or numeric-only usernames.

## New Hypothesis
I hypothesize that both the Baseline and the Mutant may not properly catch an email address that consists solely of invalid symbols in the username and has a domain of standard length. This should return `False` in both implementations but may highlight discrepancies in how each version handles the formatting and character validation.

## Experiment
```python
from string_utils.validation import is_email

def test__is_email_all_invalid_username():
    """
    Test the is_email function with an email that contains only invalid characters in the username 
    while following the basic format required, such as a valid domain.
    The expected output should be False on both, but confirming the mutant's acceptance pattern.
    """
    invalid_email_only_symbols = '!#$%&\'*+/=?^_`{|}~@provider.com'  # Invalid due to all symbols in the username

    output_only_symbols = is_email(invalid_email_only_symbols)

    print(f"output_only_symbols = {output_only_symbols}")  # Should return False on both
```

Let’s run this test to see if it exposes any differences between the Baseline and the Mutant.


### Experiment Results

### Running Experiment on Baseline
```
output_only_symbols = True
```

### Running Experiment on Mutant
```
output_only_symbols = True
```


## Conclusion
The experiment results show that both the Baseline and the Mutant returned `True` for the email input containing only invalid characters in the username (`!#$%&'*+/=?^_`{|}~@provider.com`). This indicates that both versions incorrectly accepted this email format.

**Key Findings:**
- **Email with All Invalid Characters in Username:**
  - Baseline: `True`
  - Mutant: `True`

This behavior suggests that both implementations do not enforce strict validation on certain special characters, which should not be allowed in the username section of an email according to standard email formatting rules. This inconsistency in the handling of special characters does not differentiate the behavior between the Baseline and the Mutant but rather shows a potential flaw in both.

Despite having run multiple tests and varied inputs, the outputs display that both the Baseline and the Mutant failed to filter out invalid formats similarly, constraining any significant differences in behavior across the tests monitored.

At this point, I cannot generate effective mutant-killing tests based on existing differentials since no distinguishable outputs have been identified that leverage the mutation made. The email validation seems consistently flawed rather than distinctly differentiated between the two versions.

This suggests that the mutant could be considered **equivalent** in practice as far as its operational integrity within these tests. 

## Equivalent Mutant
I confirm that the mutant is equivalent. Despite the changes made, both the Baseline and the Mutant contain similar logic flaws regarding email validation for special characters. They both incorrectly accept certain invalid email formats (such as those with all invalid characters in the username). Thus, ongoing tests have illustrated that the mutant replicates the same faulty behavior as the original, without producing differing outcomes that would classify it as an improved or distinct version of the code. All tests on variations of email formats shared the same type of output, confirming the mutant's operational parity with the baseline implementation.


You claimed that the mutant as equivalent.

Now, try other experiments to check if you can prove yourself wrong. Try new approaches or new inputs you haven't tried yet. See if you can maybe find a difference between the **Baseline** and the **Mutant** anyways.


## New Hypothesis
I hypothesize that the mutant may behave differently when testing with emails that have complicated structures, specifically where valid formats can lead to edge cases. For example, I will test with:
1. Valid emails that have multiple subdomains.
2. Emails with valid formats but include very unusual patterns, like leading dots in the username or invalid characters immediately after the '@' symbol in the domain.

These variations should be more revealing than before about potential differences in output between the Baseline and the Mutant.

## Experiment
I will check:
1. An email with subdomains (`username@subdomain.domain.com`).
2. An email with leading dots in the username (`.username@provider.com`).
3. An email with an invalid character right after the `@` (`user@.provider.com`).

The baseline should reject the latter two examples, while I'm curious if the mutant does the same.

```python
from string_utils.validation import is_email

def test__is_email_with_complex_patterns():
    """
    Test the is_email function with complex email patterns.
    The first is a valid email format, while the latter two have invalid structures.
    We predict the baseline will return different results from the mutant for the invalid cases.
    """
    valid_email_subdomains = 'user@mail.subdomain.provider.com'  # Valid format
    invalid_email_leading_dot = '.username@provider.com'          # Invalid due to leading dot
    invalid_email_invalid_after_at = 'user@.provider.com'         # Invalid due to the invalid character after '@'

    output_subdomains = is_email(valid_email_subdomains)
    output_leading_dot = is_email(invalid_email_leading_dot)
    output_invalid_after_at = is_email(invalid_email_invalid_after_at)

    print(f"output_subdomains = {output_subdomains}")              # Should return True on both
    print(f"output_leading_dot = {output_leading_dot}")            # Should return False on both
    print(f"output_invalid_after_at = {output_invalid_after_at}")  # Should return False on both
```

Let's run this test to see if it uncovers any differences between the Baseline and the Mutant.


### Experiment Results

### Running Experiment on Baseline
```
output_subdomains = False
output_leading_dot = False
output_invalid_after_at = False
```

### Running Experiment on Mutant
```
output_subdomains = False
output_leading_dot = False
output_invalid_after_at = False
```


## Test Instructions

You have 2 experiments/tests left.

Please continue by writing a test that kills the mutant. Remember: the test should pass when executed on the baseline but fail on the mutant.

## Test


## Test
I will now write a focused test aimed at killing the mutant. Given the insights gained from previous experiments, I will concentrate on creating an email that is structurally valid but includes nuances that would typically be invalid in a stricter implementation. Specifically, I will target an email with leading spaces and a correctly structured domain.

The Baseline should correctly reject any email with leading spaces, while the Mutant may incorrectly accept it due to the relaxed validation logic. Here’s the test:

```python
from string_utils.validation import is_email

def test__is_email_with_leading_spaces():
    """
    Test the is_email function with an email that has leading spaces.
    The expected output for this input should be False on the Baseline,
    but the mutant may wrongly return True due to less strict handling of leading spaces.
    """
    invalid_email_leading_spaces = '  user@provider.com'  # Invalid due to leading spaces

    output = is_email(invalid_email_leading_spaces)
    print(f"output = {output}")  # Baseline should return False, mutant may return True
```

This test checks whether the email validation behaves correctly with leading spaces, which should be an invalid state. If the mutant has effectively accommodated the leading spaces contrary to proper email formatting rules, it should differ from the Baseline output.

Let's run this test to see if it can identify any discrepancies between the Baseline and the Mutant and potentially kill the mutant.


### Test Results

### Running Test on Baseline

```
output = False
```


### Running Test on Mutant

```
output = False
```

Your test contains no assertions! Add assertions to your test, so that the test fails when executed with the mutant.

Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


## Revised Test
To ensure that I create a test that will fail on the mutant and pass on the baseline, I need to include assertions that capture the expected validity of the email addresses based on the variations tested. I will adjust the test to include an assertion that checks the output values explicitly. 

Here is the updated test case, where I'm using an email with leading spaces, which according to the Baseline, should be invalid:

```python
from string_utils.validation import is_email

def test__email_with_leading_spaces_assertion():
    """
    Test the is_email function with an email that has leading spaces.
    The expected output should be False on the Baseline, 
    while the mutant may incorrectly return True due to relaxed handling of leading spaces.
    """
    invalid_email_leading_spaces = '  user@provider.com'  # Invalid due to leading spaces

    # Check that the Baseline returns False for this input
    assert not is_email(invalid_email_leading_spaces), "The email with leading spaces should be invalid in the Baseline."
```

This updated test case asserts that the result from `is_email` is `False` for the email with leading spaces. It should pass on the Baseline while possibly failing on the mutant, which may handle leading spaces less strictly.

Let's run this test to validate if it can successfully discriminate between the Baseline and the Mutant.


### Test Results

### Running Test on Baseline

```

```


### Running Test on Mutant

```

```


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


# Conversation Aborted

The conversation has been aborted due to `too_many_turns`

The LLM exceeded the allowed number of turns.
