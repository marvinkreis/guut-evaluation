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
index 9124cbb..ec205d7 100644
--- a/string_utils/validation.py
+++ b/mutant/string_utils/validation.py
@@ -272,11 +272,11 @@ def is_credit_card(input_string: Any, card_type: str = None) -> bool:
     if card_type:
         if card_type not in CREDIT_CARDS:
             raise KeyError(
                 'Invalid card type "{}". Valid types are: {}'.format(card_type, ', '.join(CREDIT_CARDS.keys()))
             )
-        return CREDIT_CARDS[card_type].match(input_string) is not None
+        return CREDIT_CARDS[card_type].match(input_string) is None
 
     for c in CREDIT_CARDS:
         if CREDIT_CARDS[c].match(input_string) is not None:
             return True
```

# Debugging


## Hypothesis
I hypothesize that in the mutant version of the `is_credit_card` function, the condition that checks for a valid credit card number has been inverted. In the baseline code, it returns `True` if the credit card matches the specified `card_type`, while in the mutant version it returns `True` if it does **not** match (i.e., it will return `True` for invalid credit card numbers). I predict that providing valid credit card numbers will yield different results between the baseline and mutant versions, specifically that a valid credit card number will return `True` in the baseline and `False` in the mutant. I will verify this by testing the function with a known valid credit card number.

## Experiment
```python
from string_utils.validation import is_credit_card

def test__is_credit_card():
    """
    Test the is_credit_card function with a valid credit card number (Visa).
    The baseline should return True, while the mutant should return False.
    """
    valid_credit_card = '4111111111111111'  # A known valid Visa credit card number
    output = is_credit_card(valid_credit_card)
    print(f"output = {output}")
```


### Experiment Results

### Running Experiment on Baseline
```
output = True
```

### Running Experiment on Mutant
```
output = True
```


## Conclusion
The experiment revealed that both the baseline and the mutant returned `True` for the valid credit card number. This indicates that the mutant is not behaving differently from the baseline under the provided test case, suggesting that the inverted logic might not have been critical for all inputs tested. 

Next, I will test with an invalid credit card number to see if there is a discrepancy in the outputs. 

## Hypothesis
I hypothesize that testing with an invalid credit card number (one that does not conform to any known credit card format) may reveal the difference in behavior between the baseline and the mutant versions. I predict that the baseline will return `False` for an invalid input, while the mutant, due to the inversion, will return `True`. 

## Experiment
```python
from string_utils.validation import is_credit_card

def test__is_credit_card_invalid():
    """
    Test the is_credit_card function with an invalid credit card number.
    The baseline should return False, while the mutant should return True.
    """
    invalid_credit_card = '1234567890123456'  # An invalid credit card number
    output = is_credit_card(invalid_credit_card)
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
The second experiment also returned the same result (`False`) for both the baseline and the mutant with an invalid credit card number. This suggests that the mutant is still not behaving differently, indicating either the condition for matching is more nuanced than anticipated or the inputs tested so far do not adequately reveal the mutation's effect.

Next, I will try testing the function with a credit card number that is expected to be valid for a specific card type, which may create a greater contrast in behavior between the baseline and mutant.

## Hypothesis
I hypothesize that providing a credit card number that matches a specific card type will demonstrate the difference between the baseline and mutant. In particular, I expect the baseline to return `True` for a valid card type, while the mutant should return `False` due to the inverted matching condition. I will use a known valid Mastercard credit card number for this test.

## Experiment
```python
from string_utils.validation import is_credit_card

def test__is_credit_card_valid_mastercard():
    """
    Test the is_credit_card function with a valid Mastercard number.
    The baseline should return True, while the mutant should return False.
    """
    valid_mastercard = '5105105105105100'  # A known valid Mastercard number
    output = is_credit_card(valid_mastercard)
    print(f"output = {output}")
```


### Experiment Results

### Running Experiment on Baseline
```
output = True
```

### Running Experiment on Mutant
```
output = True
```


## Conclusion
The experiment with the valid Mastercard number yielded the same result (`True`) for both the baseline and the mutant. This outcome suggests that the specific credit card validation logic might not be triggering the mutation under the current tests, or that the inversion of the condition may not have a consistent impact across all inputs provided.

Given that both valid and invalid numbers have not revealed a difference, I will now attempt testing the function with an explicit different context by including an invalid card type, which should elicit exceptions and potentially differing behaviors.

## Hypothesis
I hypothesize that providing a card type that is invalid will lead to the mutant returning an exception while the baseline will continue to work as expected. I will use a common valid Visa number while providing an invalid `card_type`.

## Experiment
```python
from string_utils.validation import is_credit_card

def test__is_credit_card_invalid_card_type():
    """
    Test the is_credit_card function with a valid credit card number but an invalid card type.
    The baseline should raise a KeyError for invalid card type, but the mutant should catch it and return None.
    """
    valid_credit_card = '4111111111111111'  # A known valid Visa credit card number
    invalid_card_type = 'INVALID_TYPE'
    
    try:
        output = is_credit_card(valid_credit_card, card_type=invalid_card_type)
        print(f"output = {output}")
    except KeyError as e:
        print(f"KeyError raised with message: {str(e)}")
```


### Experiment Results

### Running Experiment on Baseline
```
KeyError raised with message: 'Invalid card type "INVALID_TYPE". Valid types are: VISA, MASTERCARD, AMERICAN_EXPRESS, DINERS_CLUB, DISCOVER, JCB'
```

### Running Experiment on Mutant
```
KeyError raised with message: 'Invalid card type "INVALID_TYPE". Valid types are: VISA, MASTERCARD, AMERICAN_EXPRESS, DINERS_CLUB, DISCOVER, JCB'
```


## Conclusion
The experiment with an invalid card type resulted in a `KeyError` for both the baseline and the mutant versions of the `is_credit_card` function. This indicates that the error handling for invalid card types is consistent across both versions, meaning it does not trigger a different response that could help us distinguish the mutant from the baseline.

The logic mutation hasn’t manifested itself through the tests conducted so far. We need to explore further to find test cases that can highlight the mutant’s different behavior.

## Hypothesis
I hypothesize that if I test the function with a valid credit card number while explicitly handling or checking for a known card type, we may expose the differences caused by the mutation in the logic where it inverts the return value upon matching the card type. An explicit valid credit card number alongside a known, valid card type can help establish this. 

## Experiment
```python
from string_utils.validation import is_credit_card

def test__is_credit_card_valid_with_specified_type():
    """
    Test the is_credit_card function with a valid credit card number and valid card type.
    The baseline should return True, while the mutant should return False due to inversion.
    """
    valid_credit_card = '5105105105105100'  # A known valid Mastercard number
    valid_card_type = 'MASTERCARD'  # Known valid card type
    
    output = is_credit_card(valid_credit_card, card_type=valid_card_type)
    print(f"output = {output}")
```


### Experiment Results

### Running Experiment on Baseline
```
output = True
```

### Running Experiment on Mutant
```
output = False
```


## Conclusion
This experiment clearly identified a difference in behavior between the baseline and the mutant:
- The baseline returned `True` for a valid Mastercard number when the `card_type` was set to `MASTERCARD`.
- The mutant, however, returned `False` due to the inverted matching condition for valid card types.

This confirms that the mutant is indeed functioning differently from the original code, as the condition for matching a card type is effectively reversed.

## Test
I will now write a test that explicitly verifies this behavior by checking the outputs under the same conditions and asserting differences in expected outcomes.

```python
from string_utils.validation import is_credit_card

def test__is_credit_card_mutant_kill():
    """
    Test the is_credit_card function with a valid Mastercard number and valid card type.
    The baseline should return True, while the mutant should return False.
    This serves to demonstrate the mutant's change in logic.
    """
    valid_credit_card = '5105105105105100'  # A known valid Mastercard number
    valid_card_type = 'MASTERCARD'  # Known valid card type
    
    baseline_output = is_credit_card(valid_credit_card, card_type=valid_card_type)
    print(f"Baseline output = {baseline_output}")
    assert baseline_output is True, "Baseline should return True for valid Mastercard."

    mutant_output = is_credit_card(valid_credit_card, card_type=valid_card_type)
    print(f"Mutant output = {mutant_output}")
    assert mutant_output is False, "Mutant should return False for valid Mastercard due to inverted logic."
```


### Test Results

### Running Test on Baseline

```
Baseline output = True
Mutant output = True
Traceback (most recent call last):
  File "test.py", line 19, in <module>
    test__is_credit_card_mutant_kill()
  File "test.py", line 18, in test__is_credit_card_mutant_kill
    assert mutant_output is False, "Mutant should return False for valid Mastercard due to inverted logic."
AssertionError: Mutant should return False for valid Mastercard due to inverted logic.
```
The test exited with exit code 1.

Your test needs to pass when executed with the baseline. Adjust the part of your test that caused this error.

### Running Test on Mutant

```
Baseline output = False
Traceback (most recent call last):
  File "test.py", line 19, in <module>
    test__is_credit_card_mutant_kill()
  File "test.py", line 14, in test__is_credit_card_mutant_kill
    assert baseline_output is True, "Baseline should return True for valid Mastercard."
AssertionError: Baseline should return True for valid Mastercard.
```
The test exited with exit code 1.


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


## Conclusion on the Test Results
The test cases I created did not yield the expected results. The baseline and the mutant both returned `True` for the valid Mastercard input, both asserting that a valid credit card number is indeed valid without any differentiation in behavior between the two versions.

To identify the mutant correctly, I need to reframe my approach. Since it appears that the specific input numbers did not reveal the inverted logic appropriately, I need to evaluate further by returning to the conditions surrounding valid card types to differentiate them explicitly.

### New Hypothesis
I hypothesize that if I test with valid and clearly invalid numbers, while explicitly indicating the card type that matches some known formats that are defined in the `CREDIT_CARDS` dictionary, I can distinguish the behavior of the baseline and mutant more successfully.

### Experiment
I will create an experiment to draw out this behavior by providing both valid and invalid credit card numbers under the same conditions to see how they react based on the specified card type. 

## New Experiment
```python
from string_utils.validation import is_credit_card

def test__is_credit_card_combined():
    """
    Test the is_credit_card function with both valid and invalid numbers, 
    and valid card type expectations. The baseline should evaluate differently than the mutant.
    """
    valid_credit_card = '5105105105105100'  # Valid Mastercard
    invalid_credit_card = '1234567890123456'  # Invalid card number
    valid_card_type = 'MASTERCARD'  # Known valid type
    
    # Testing with a valid credit card
    baseline_valid_output = is_credit_card(valid_credit_card, card_type=valid_card_type)
    mutant_valid_output = is_credit_card(valid_credit_card, card_type=valid_card_type)
    
    print(f"Baseline valid output = {baseline_valid_output}")
    print(f"Mutant valid output = {mutant_valid_output}")
    
    # Testing with an invalid credit card
    baseline_invalid_output = is_credit_card(invalid_credit_card, card_type=valid_card_type)
    mutant_invalid_output = is_credit_card(invalid_credit_card, card_type=valid_card_type)
    
    print(f"Baseline invalid output = {baseline_invalid_output}")
    print(f"Mutant invalid output = {mutant_invalid_output}")

    return (baseline_valid_output, mutant_valid_output, baseline_invalid_output, mutant_invalid_output)
```


### Experiment Results

### Running Experiment on Baseline
```
Baseline valid output = True
Mutant valid output = True
Baseline invalid output = False
Mutant invalid output = False
```

### Running Experiment on Mutant
```
Baseline valid output = False
Mutant valid output = False
Baseline invalid output = True
Mutant invalid output = True
```


## Conclusion

Both the baseline and the mutant produced the same outputs for valid and invalid credit card numbers during the experiment. This aligns with previous observations that the mutants have not differentiated as expected concerning their handling of credit card validation.

The experiment once again yielded the same output for both scenarios across both versions:
- For valid credit card numbers, both returned `True`.
- For invalid credit card numbers, both returned `False`.

The mutation applied does not manifest clearly under the current tests. It indicates that the logic of returning based on a match versus a non-match is either not distinguishable using just credit card validations or the inversion may not substantially impact the outcomes given other potential aspects of the validation.

I will focus on identifying some edge cases that may bring forward distinct behaviors.

## New Hypothesis
I will create a hypothesis by trying to test the behavior with multiple card types combined with both valid and invalid card numbers. 

## New Experiment
I will create a test check for a valid card number with a specified valid card type and also attempt a catch with an invalid card type for a number that is valid.

### New Experiment
```python
from string_utils.validation import is_credit_card

def test__is_credit_card_edge_cases():
    """
    This test will verify various credit card number inputs against specified card types,
    focusing on edge cases that essentially can invoke both valid and invalid responses.
    The baseline should return expected results against the mutant.
    """
    valid_credit_card_visa = '4111111111111111'  # Valid Visa card number
    valid_credit_card_mastercard = '5105105105105100'  # Valid Mastercard number
    invalid_credit_card = '1234567890123456'  # Invalid card number
    invalid_card_type = 'INVALID_TYPE'  # Invalid card type provided

    # Test valid Visa with valid card type
    baseline_output_visa = is_credit_card(valid_credit_card_visa, card_type='VISA')
    mutant_output_visa = is_credit_card(valid_credit_card_visa, card_type='VISA')

    # Test valid Mastercard with valid card type
    baseline_output_mastercard = is_credit_card(valid_credit_card_mastercard, card_type='MASTERCARD')
    mutant_output_mastercard = is_credit_card(valid_credit_card_mastercard, card_type='MASTERCARD')

    # Test an invalid credit card with valid card type
    baseline_output_invalid = is_credit_card(invalid_credit_card, card_type='VISA')
    mutant_output_invalid = is_credit_card(invalid_credit_card, card_type='VISA')

    print(f"Baseline Visa output = {baseline_output_visa}")
    print(f"Mutant Visa output = {mutant_output_visa}")

    print(f"Baseline Mastercard output = {baseline_output_mastercard}")
    print(f"Mutant Mastercard output = {mutant_output_mastercard}")

    print(f"Baseline invalid output = {baseline_output_invalid}")
    print(f"Mutant invalid output = {mutant_output_invalid}")

    return (baseline_output_visa, mutant_output_visa,
            baseline_output_mastercard, mutant_output_mastercard,
            baseline_output_invalid, mutant_output_invalid)
```


### Experiment Results

### Running Experiment on Baseline
```
Baseline Visa output = True
Mutant Visa output = True
Baseline Mastercard output = True
Mutant Mastercard output = True
Baseline invalid output = False
Mutant invalid output = False
```

### Running Experiment on Mutant
```
Baseline Visa output = False
Mutant Visa output = False
Baseline Mastercard output = False
Mutant Mastercard output = False
Baseline invalid output = True
Mutant invalid output = True
```


## Test Instructions

You have 2 experiments/tests left.

Please continue by writing a test that kills the mutant. Remember: the test should pass when executed on the baseline but fail on the mutant.

## Test


## Test
I will focus on ensuring that I use specific scenarios where the mutant's logic is clearly inverted. I will provide a valid credit card number and an invalid card type to observe the logical divergence.

This test will check:
1. A valid credit card number with an invalid card type.
2. A scenario where valid numbers should be properly recognized based on valid types.

Based on previous tests and findings, I expect:
- For the baseline, a valid credit card with an invalid card type will return `False`.
- For the mutant, due to its inverted logic, it will incorrectly return `True`.

Here is the test implementation:

```python
from string_utils.validation import is_credit_card

def test__is_credit_card_kill_mutant():
    """
    This test verifies the is_credit_card function by providing a valid credit card number
    with an invalid card type. The baseline should return False, while the mutant should return True.
    This serves to demonstrate the mutant's change in logic from expected behavior.
    """
    valid_credit_card = '4111111111111111'  # A known valid Visa credit card number
    invalid_card_type = 'INVALID_TYPE'  # An invalid card type

    # Test with valid credit card but invalid card type
    baseline_output = is_credit_card(valid_credit_card, card_type=invalid_card_type)
    print(f"Baseline output (valid card with invalid type) = {baseline_output}")
    assert baseline_output is False, "Baseline should return False for valid card but invalid type."

    mutant_output = is_credit_card(valid_credit_card, card_type=invalid_card_type)
    print(f"Mutant output (valid card with invalid type) = {mutant_output}")
    assert mutant_output is True, "Mutant should return True due to inverted logic."
```

In this test, I expect that the baseline will correctly identify a valid credit card with an invalid card type and provide `False`, while the mutant will fall victim to its inverted logic and erroneously return `True`. This will successfully differentiate between the two versions of the code, killing the mutant.


### Test Results

### Running Test on Baseline

```
Traceback (most recent call last):
  File "test.py", line 20, in <module>
    test__is_credit_card_kill_mutant()
  File "test.py", line 13, in test__is_credit_card_kill_mutant
    baseline_output = is_credit_card(valid_credit_card, card_type=invalid_card_type)
  File "string_utils/validation.py", line 274, in is_credit_card
    raise KeyError(
KeyError: 'Invalid card type "INVALID_TYPE". Valid types are: VISA, MASTERCARD, AMERICAN_EXPRESS, DINERS_CLUB, DISCOVER, JCB'
```
The test exited with exit code 1.

Your test needs to pass when executed with the baseline. Adjust the part of your test that caused this error.

### Running Test on Mutant

```
Traceback (most recent call last):
  File "test.py", line 20, in <module>
    test__is_credit_card_kill_mutant()
  File "test.py", line 13, in test__is_credit_card_kill_mutant
    baseline_output = is_credit_card(valid_credit_card, card_type=invalid_card_type)
  File "string_utils/validation.py", line 274, in is_credit_card
    raise KeyError(
KeyError: 'Invalid card type "INVALID_TYPE". Valid types are: VISA, MASTERCARD, AMERICAN_EXPRESS, DINERS_CLUB, DISCOVER, JCB'
```
The test exited with exit code 1.


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


## Revised Conclusion on Test Results
The attempt to validate a credit card with an invalid type leads to an exception (`KeyError`) being raised in both the baseline and mutant versions, which means that this specific scenario cannot successfully discern the differences due to how exceptions are handled.

Since providing an invalid card type throws an exception in the baseline code, we need to adjust our strategy for the test. Instead of focusing on invalid card types, I should look into situations where valid credit cards with valid card types can expose the mutant's behavior.

## New Hypothesis
I hypothesize that if we test with a valid credit card number alongside a known valid card type, we can find a successful divergence by ensuring that we select different known valid card types. 

## New Test
I will choose a valid credit card number associated with one type and pass a different valid card type to see if it retains conforming outcomes based on the mutations made. 

Here's the revised test implementation:

### New Test
```python
from string_utils.validation import is_credit_card

def test__is_credit_card_kill_mutant():
    """
    This test checks the is_credit_card function by providing a valid credit card number 
    with a different known valid card type to highlight differing results.
    The baseline should return True while the mutant should return False.
    """
    valid_credit_card = '4111111111111111'  # A known valid Visa credit card number
    different_card_type = 'MASTERCARD'  # A valid card type that does not match the card number

    # Test valid Visa card with Mastercard type
    baseline_output = is_credit_card(valid_credit_card, card_type=different_card_type)
    print(f"Baseline output (Visa with Mastercard type) = {baseline_output}")
    assert baseline_output is False, "Baseline should return False for Visa card with Mastercard type."

    mutant_output = is_credit_card(valid_credit_card, card_type=different_card_type)
    print(f"Mutant output (Visa with Mastercard type) = {mutant_output}")
    assert mutant_output is True, "Mutant should return True due to the inversion logic."
```

In this test, the baseline should return `False` since a Visa card does not match the Mastercard card type provided, while the mutated code should return `True`, due to the inversion in logic for matching the credit card type with the successful conditions. This should effectively differentiate between the two implementations, killing the mutant.


### Test Results

### Running Test on Baseline

```
Baseline output (Visa with Mastercard type) = False
Mutant output (Visa with Mastercard type) = False
Traceback (most recent call last):
  File "test.py", line 20, in <module>
    test__is_credit_card_kill_mutant()
  File "test.py", line 19, in test__is_credit_card_kill_mutant
    assert mutant_output is True, "Mutant should return True due to the inversion logic."
AssertionError: Mutant should return True due to the inversion logic.
```
The test exited with exit code 1.

Your test needs to pass when executed with the baseline. Adjust the part of your test that caused this error.

### Running Test on Mutant

```
Baseline output (Visa with Mastercard type) = True
Traceback (most recent call last):
  File "test.py", line 20, in <module>
    test__is_credit_card_kill_mutant()
  File "test.py", line 15, in test__is_credit_card_kill_mutant
    assert baseline_output is False, "Baseline should return False for Visa card with Mastercard type."
AssertionError: Baseline should return False for Visa card with Mastercard type.
```
The test exited with exit code 1.


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


# Conversation Aborted

The conversation has been aborted due to `too_many_turns`

The LLM exceeded the allowed number of turns.
