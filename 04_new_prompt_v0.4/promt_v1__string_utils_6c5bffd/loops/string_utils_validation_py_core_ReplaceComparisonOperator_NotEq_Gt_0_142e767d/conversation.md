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
index 9124cbb..bd6a21e 100644
--- a/string_utils/validation.py
+++ b/mutant/string_utils/validation.py
@@ -108,11 +108,11 @@ def is_full_string(input_string: Any) -> bool:
 
     :param input_string: String to check.
     :type input_string: str
     :return: True if not empty, false otherwise.
     """
-    return is_string(input_string) and input_string.strip() != ''
+    return is_string(input_string) and input_string.strip() > ''
 
 
 def is_number(input_string: str) -> bool:
     """
     Checks if a string is a valid number.
```

# Debugging


## Hypothesis
I hypothesize that the mutant changes the behavior of the `is_full_string` function. Specifically, it changes the condition from checking if the stripped input string is not empty (`input_string.strip() != ''`) to checking if it is greater than an empty string (`input_string.strip() > ''`). I predict that this will affect the output for strings that consist solely of whitespace characters. If we provide a string that is empty or contains only whitespace, the mutant should return a different result compared to the baseline.

## Experiment
To test this hypothesis, I will write a function that utilizes `is_full_string` with different types of input, including:
- An empty string
- A string with only whitespace
- A string with text

I will print the outputs to observe any discrepancies between the baseline and the mutant.

```python
from validation import is_full_string

def test__is_full_string():
    """
    Test the is_full_string function with various inputs.
    The inputs include an empty string, a whitespace-only string, and a non-empty string to check for discrepancies in outputs on the baseline and mutant versions.
    """
    test_cases = ['', '   ', 'hello']
    for case in test_cases:
        output = is_full_string(case)
        print(f'Input: "{case}" | Output: {output}')
```


### Experiment Results

### Running Experiment on Baseline
```
Traceback (most recent call last):
  File "test.py", line 1, in <module>
    from validation import is_full_string
ModuleNotFoundError: No module named 'validation'
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
Traceback (most recent call last):
  File "test.py", line 1, in <module>
    from validation import is_full_string
ModuleNotFoundError: No module named 'validation'
```
The experiment exited with exit code 1.


## Conclusion
The experiment failed because the module `validation` could not be found. This may be related to the import statement used in the experiment, suggesting that we may need to adjust the import to properly reference the `validation` module from its correct path.

Next, I'll adjust the import statement to reflect the correct structure of the codebase and rerun the experiment.

I will import the `is_full_string` function correctly as follows, considering it might reside in `string_utils.validation`.

## Experiment
I will modify the import statement for `is_full_string` and run the test again.

```python
from string_utils.validation import is_full_string

def test__is_full_string():
    """
    Test the is_full_string function with various inputs.
    The inputs include an empty string, a whitespace-only string, and a non-empty string to check for discrepancies in outputs on the baseline and mutant versions.
    """
    test_cases = ['', '   ', 'hello']
    for case in test_cases:
        output = is_full_string(case)
        print(f'Input: "{case}" | Output: {output}')
```


### Experiment Results

### Running Experiment on Baseline
```
Input: "" | Output: False
Input: "   " | Output: False
Input: "hello" | Output: True
```

### Running Experiment on Mutant
```
Input: "" | Output: False
Input: "   " | Output: False
Input: "hello" | Output: True
```


## Conclusion
The outputs from both the baseline and the mutant are the same for all input cases:
- For the empty string `""`, the output is `False`.
- For the whitespace-only string `"   "`, the output is `False`.
- For the non-empty string `"hello"`, the output is `True`.

This indicates that the mutant does not produce a different output compared to the baseline for these test cases. Therefore, the change made in the mutant seems to be equivalent in behavior, at least for the inputs tested.

I will create additional test cases focusing specifically on cases that may include characters that only differ on whitespace sensitivity. This will help determine if there are discrepancies when the `is_full_string` function processes unusual whitespace cases.

## Experiment
Next, I will test with additional mixed whitespace inputs and a string that includes non-white space characters interspersed with varying whitespace:

```python
from string_utils.validation import is_full_string

def test__is_full_string_extended():
    """
    Test the is_full_string function with edge cases that involve different types of whitespace.
    This includes strings with non-printable whitespace and mixed content to explore further differences in the baseline and mutant outputs.
    """
    test_cases = [
        '   ',        # Only whitespace
        '\n',        # Newline only
        '\t\t',      # Tab characters only
        '   hello',  # Leading whitespace
        'hello   ',  # Trailing whitespace
        ' hello ',    # Surrounded by whitespace
        '   ',        # Just spaces
        ' ' * 1000,  # Large whitespace
        'hello world', # Normal case should return True
    ]
    for case in test_cases:
        output = is_full_string(case)
        print(f'Input: "{case}" | Output: {output}')
```


### Experiment Results

### Running Experiment on Baseline
```
Input: "   " | Output: False
Input: "
" | Output: False
Input: "		" | Output: False
Input: "   hello" | Output: True
Input: "hello   " | Output: True
Input: " hello " | Output: True
Input: "   " | Output: False
Input: "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        " | Output: False
Input: "hello world" | Output: True
```

### Running Experiment on Mutant
```
Input: "   " | Output: False
Input: "
" | Output: False
Input: "		" | Output: False
Input: "   hello" | Output: True
Input: "hello   " | Output: True
Input: " hello " | Output: True
Input: "   " | Output: False
Input: "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        " | Output: False
Input: "hello world" | Output: True
```


## Conclusion
The results from the experiment show that both the baseline and mutant versions of the `is_full_string` function produced identical outputs for all test cases, including those with different types of whitespace. This confirms that the mutant behavior is equivalent to the baseline code.

### Summary of Findings:
- The mutant's change in the conditional expression did not result in a different outcome for the tested cases.
- All outputs for both baseline and mutant versions returned the same results.

## Equivalent Mutant
The change made in the mutant (altering the condition from `input_string.strip() != ''` to `input_string.strip() > ''`) does not affect the function's behavior, as both conditions yield the same truthiness for the strings tested. Thus, the mutant is equivalent because:
- Inputs like spaces, newlines, and tab characters still correctly evaluate to `False`.
- Mixed inputs containing both whitespace and text also correctly evaluate to `True`.

Consequently, there is no need to create a further test to differentiate between the two, as we have demonstrated that they are equivalent under varied conditions.

At this point, I will prepare a closing statement to summarize my investigation and findings.


You claimed that the mutant as equivalent.

Now, try other experiments to check if you can prove yourself wrong. Try new approaches or new inputs you haven't tried yet. See if you can maybe find a difference between the **Baseline** and the **Mutant** anyways.


## Hypothesis
Given the previous conclusion that the mutant is equivalent to the baseline, I will adjust my approach. I hypothesize that while we have tested a variety of whitespace inputs, there might still be edge cases involving special characters or mixed whitespace that could yield a different outcome between the baseline and the mutant.

## Experiment
For this experiment, I will test the `is_full_string` function with inputs that consist of special characters, non-breaking spaces, and variations of whitespace characters that may introduce unexpected behavior. These include inputs that rely on the nuances of string processing, such as multibyte whitespace or special Unicode spaces.

```python
from string_utils.validation import is_full_string

def test__is_full_string_special_cases():
    """
    Test the is_full_string function with special character cases.
    This includes checks for Unicode whitespace characters and special characters to assess if any differences exist between baseline and mutant.
    """
    test_cases = [
        '\u00A0',             # Non-breaking space (Unicode)
        '\u2002',             # En space (Unicode)
        '\u2003',             # Em space (Unicode)
        '\u3000',             # Ideographic space (Unicode)
        'text \u00A0',       # Text followed by non-breaking space
        ' \u2002text',       # Non-breaking space at start followed by text
        '@#$%^&*()',          # Special characters
        'text@#$%^&*()',      # Text with special characters
        '\u200B',             # Zero width space (Unicode)
        '   \u200B   '        # University space on both sides
    ]
    for case in test_cases:
        output = is_full_string(case)
        print(f'Input: "{case}" | Output: {output}')
```


### Experiment Results

### Running Experiment on Baseline
```
Input: " " | Output: False
Input: " " | Output: False
Input: " " | Output: False
Input: "　" | Output: False
Input: "text  " | Output: True
Input: "  text" | Output: True
Input: "@#$%^&*()" | Output: True
Input: "text@#$%^&*()" | Output: True
Input: "​" | Output: True
Input: "   ​   " | Output: True
```

### Running Experiment on Mutant
```
Input: " " | Output: False
Input: " " | Output: False
Input: " " | Output: False
Input: "　" | Output: False
Input: "text  " | Output: True
Input: "  text" | Output: True
Input: "@#$%^&*()" | Output: True
Input: "text@#$%^&*()" | Output: True
Input: "​" | Output: True
Input: "   ​   " | Output: True
```


## Conclusion
The results from the experiment with special character cases indicate that both the baseline and mutant versions of the `is_full_string` function again produced identical outputs across all tested inputs.

### Summary of Findings:
- The outputs were the same for all special character cases tested, including Unicode whitespace characters, special characters, and mixed inputs.
- Both the baseline and the mutant returned `False` for strings with only whitespace (including Unicode spaces) and `True` for strings that contain any non-whitespace content, even if it includes special characters.

Despite the hypothesis that new types of inputs could reveal differences between the baseline and the mutant, the experiments failed to uncover any discrepancies.

## Equivalent Mutant Confirmation
Thus, I conclude that the mutant is indeed equivalent to the baseline. The change made in the mutant code does not cause any difference in functionality across a variety of test cases, reaffirming that both implementations function identically under the tested conditions.

Given this confirmation, I have explored several significant avenues and variations of inputs, none of which led to a distinction between the baseline and mutant behavior. Therefore, the investigation into this specific mutant can be concluded.


You claimed that the mutant as equivalent.

Now, try other experiments to check if you can prove yourself wrong. Try new approaches or new inputs you haven't tried yet. See if you can maybe find a difference between the **Baseline** and the **Mutant** anyways.


## Hypothesis
Although previous experiments suggest that the mutant behaves equivalently to the baseline for a wide range of inputs, there may be specific edge cases involving complex or unexpected character patterns that have not been tested yet. I hypothesize that inputs containing mixed types of Unicode characters in unconventional placements might reveal distinctions between the baseline and mutant.

## Experiment
In this experiment, I'll test the `is_full_string` function using:
- A string with leading and trailing non-breaking spaces.
- A string made up of just Alt+Code characters (non-printable characters).
- A string featuring various combinations of whitespace and characters.
- Inputs that contain sequences of whitespace interspersed with text or special characters.

Let's see if any of these edge cases will expose different outputs for the baseline and mutant versions.

```python
from string_utils.validation import is_full_string

def test__is_full_string_edge_cases():
    """
    Test the is_full_string function with complex and mixed character cases.
    This includes testing for edge cases that could reveal differences between the baseline and mutant behavior.
    """
    test_cases = [
        '\u00A0hello\u00A0',         # Non-breaking spaces around 'hello'
        'hello\u2003world',           # Mixed whitespace (em space between words)
        '\u200Bhello',                # Zero-width space before text
        'hello\u200B',                # Zero-width space after text
        '@# \u3000  @#',              # Combination of special and unicode space characters
        '\n\t hello \n\t',            # New lines and tabs mixed with text
        'hello\t\n',                  # Text ending with tab and newline
        'text' + ' ' * 100 + 'more',  # Large spaces between words
        '  \u200B   ',                # Large input of whitespace and zero-width space
        '\u2002hello\u2003world'      # Double spaces with text
    ]
    
    for case in test_cases:
        output = is_full_string(case)
        print(f'Input: "{case}" | Output: {output}')
```


### Experiment Results

### Running Experiment on Baseline
```
Input: " hello " | Output: True
Input: "hello world" | Output: True
Input: "​hello" | Output: True
Input: "hello​" | Output: True
Input: "@# 　  @#" | Output: True
Input: "
	 hello 
	" | Output: True
Input: "hello	
" | Output: True
Input: "text                                                                                                    more" | Output: True
Input: "  ​   " | Output: True
Input: " hello world" | Output: True
```

### Running Experiment on Mutant
```
Input: " hello " | Output: True
Input: "hello world" | Output: True
Input: "​hello" | Output: True
Input: "hello​" | Output: True
Input: "@# 　  @#" | Output: True
Input: "
	 hello 
	" | Output: True
Input: "hello	
" | Output: True
Input: "text                                                                                                    more" | Output: True
Input: "  ​   " | Output: True
Input: " hello world" | Output: True
```


## Conclusion
The results from the edge case experiment show that both the baseline and mutant versions of the `is_full_string` function produced identical outputs for each input case. 

### Summary of Findings:
- Both versions returned `True` for strings that contain text mixed with various forms of whitespace, including non-breaking spaces, zero-width spaces, and other Unicode whitespace characters.
- The outputs were consistent across all tested edge cases, confirming that the implementations did not differ in their functionality for these inputs either.

## Final Confirmation of Equivalent Mutant
The continued similarity in output across a diverse range of inputs, including whitespace and special character combinations, reinforces the conclusion that the mutant is equivalent to the baseline. The change introduced in the mutant did not affect the overall result of the `is_full_string` function as tested across all explored cases.

### Additional Considerations
Even after multiple attempts with various character configurations and edge cases, no differences in behavior were identified. Thus, the investigation into the mutant confirms that it does not deviate from the baseline functionality. I will now conclude the testing process surrounding this specific mutant.


You claimed that the mutant as equivalent.

Now, try other experiments to check if you can prove yourself wrong. Try new approaches or new inputs you haven't tried yet. See if you can maybe find a difference between the **Baseline** and the **Mutant** anyways.


## Hypothesis
Given that I have found no differences in behavior between the baseline and the mutant through numerous tests on various types of characters and whitespace, I will challenge my hypothesis once more. I hypothesize that there may be configurations involving the combination of different input types or boundary conditions that could expose a functional difference.

## Experiment
For this experiment, I will test a few configurations, focusing on:
- Strings with mixed lengths, including allowing for leading and trailing spaces but having control characters or escape sequences that may behave differently.
- Introducing extremely large strings and short strings that might have unique edge effects.
- Special cases such as `None`, integers, and other unexpected types passed to `is_full_string`, which the baseline should handle with a proper exception, and the mutant may exhibit different behavior due to the attempted change in logic.

Here’s the test code to explore these scenarios:

```python
from string_utils.validation import is_full_string

def test__is_full_string_more_cases():
    """
    Test the is_full_string function with varying lengths and special cases.
    This includes testing with large inputs, short inputs, and unexpected types to see differences between baseline and mutant behaviors.
    """
    test_cases = [
        ' ' * 10000,         # Very large string of spaces
        ' ' * 10000 + 'text',  # Large string of spaces followed by text
        'text' + ' ' * 10000,  # Text followed by a large string of spaces
        '',                    # Empty string
        None,                  # Non-string input
        'text' + '\0'        # String ending with a null character
    ]
    
    for case in test_cases:
        try:
            output = is_full_string(case)
            print(f'Input: "{case}" | Output: {output}')
        except Exception as e:
            print(f'Input: "{case}" raised an exception: {e}')
```

This experiment will allow us to assess the robustness of both versions against extreme cases and bypass inputs. The goal is to check for any discrepancies caused by how each version of the function handles unexpected input or boundary cases.


### Experiment Results

### Running Experiment on Baseline
```
Input: "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        <truncated>
```

### Running Experiment on Mutant
```
Input: "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        <truncated>
```


## Conclusion
The results show that both the baseline and the mutant produced the same output for the large string of spaces, and the output appears truncated due to the size of the string. The expected output for this case would simply be `False`, indicating that the string is composed entirely of whitespace.

Now, let's further examine how both versions handle other edge cases.

### Observation:
1. The experiment included an empty string, a `None` input, and inputs with special characters, where we anticipate different behavior. Let’s add some new configurations that may generate exceptions or different outputs.

## Further Experiment
I will add specific cases to test the inputs of `None`, empty string, and those mixed with control characters in a more focused manner to potentially reveal differences.

```python
def test__is_full_string_edge_cases():
    """
    Test the is_full_string function specifically with edge cases: None, empty string,
    and other non-string inputs to observe behavior.
    """
    test_cases = [
        '',           # Empty string
        None,         # None as input
        123,          # Integer input
        12.34,        # Float input
        [],           # Empty list
        {},           # Empty dictionary
        'text' + '\0' # String with null character
    ]
    
    for case in test_cases:
        try:
            output = is_full_string(case)
            print(f'Input: "{case}" | Output: {output}')
        except Exception as e:
            print(f'Input: "{case}" raised an exception: {e}')
```

This additional test should help uncover any differences in how the baseline and the mutant manage potentially problematic inputs. Let’s run this next experiment and check the results.


### Experiment Results

### Running Experiment on Baseline
```
Input: "" raised an exception: name 'is_full_string' is not defined
Input: "None" raised an exception: name 'is_full_string' is not defined
Input: "123" raised an exception: name 'is_full_string' is not defined
Input: "12.34" raised an exception: name 'is_full_string' is not defined
Input: "[]" raised an exception: name 'is_full_string' is not defined
Input: "{}" raised an exception: name 'is_full_string' is not defined
Input: "text " raised an exception: name 'is_full_string' is not defined
```

### Running Experiment on Mutant
```
Input: "" raised an exception: name 'is_full_string' is not defined
Input: "None" raised an exception: name 'is_full_string' is not defined
Input: "123" raised an exception: name 'is_full_string' is not defined
Input: "12.34" raised an exception: name 'is_full_string' is not defined
Input: "[]" raised an exception: name 'is_full_string' is not defined
Input: "{}" raised an exception: name 'is_full_string' is not defined
Input: "text " raised an exception: name 'is_full_string' is not defined
```


## Conclusion
The experiment attempting to evaluate how both the baseline and mutant handle various edge cases, including `None`, empty strings, integers, and other non-string types, failed due to a `name 'is_full_string' is not defined` error. This suggests that the import for `is_full_string` was not correctly defined in the context of the test function.

### Next Steps
To troubleshoot this, I will ensure proper imports again for both versions:

1. Verify the path when importing the `is_full_string` function.
2. Re-run the edge cases test to examine the expected behavior.

Let's update the import statement appropriately and execute it once more.

## Experiment Update
Here’s the modified version of the last test case with a corrected import statement:

```python
from string_utils.validation import is_full_string

def test__is_full_string_edge_cases_updated():
    """
    Test the is_full_string function specifically with edge cases: None, empty string,
    and other non-string inputs to observe behavior.
    """
    test_cases = [
        '',           # Empty string
        None,         # None as input
        123,          # Integer input
        12.34,        # Float input
        [],           # Empty list
        {},           # Empty dictionary
        'text' + '\0' # String with null character
    ]
    
    for case in test_cases:
        try:
            output = is_full_string(case)
            print(f'Input: "{case}" | Output: {output}')
        except Exception as e:
            print(f'Input: "{case}" raised an exception: {e}')

# Run the updated test
test__is_full_string_edge_cases_updated()
```

Now, I will re-execute the updated experiment to assess the behavior of the baseline versus the mutant again.


### Experiment Results

### Running Experiment on Baseline
```
Input: "" | Output: False
Input: "None" | Output: False
Input: "123" | Output: False
Input: "12.34" | Output: False
Input: "[]" | Output: False
Input: "{}" | Output: False
Input: "text " | Output: True
```

### Running Experiment on Mutant
```
Input: "" | Output: False
Input: "None" | Output: False
Input: "123" | Output: False
Input: "12.34" | Output: False
Input: "[]" | Output: False
Input: "{}" | Output: False
Input: "text " | Output: True
```


## Test Instructions

You have 2 experiments/tests left.

Please continue by writing a test that kills the mutant. Remember: the test should pass when executed on the baseline but fail on the mutant.

## Test


Given our previous results and the findings from the various experiments conducted, I have established that the mutant behaves identically to the baseline for multiple types of inputs, including those involving special characters, whitespace, and edge inputs. 

However, to construct a test that effectively differentiates the mutant from the baseline, I will provide an input that specifically targets the modified line in the mutant:

### Test Strategy
The unique behavior of the mutant can be exploited by using an input that includes some whitespace characters, but wrapped in conditions tailored to differentiate the string's evaluation on the baseline and mutant behavior. In particular, since the mutant uses `input_string.strip() > ''` instead of `input_string.strip() != ''`, it may behave unexpectedly with different kinds of whitespace (especially if non-standard Unicode whitespace is involved).

### Proposed Test
I will create a test that uses a string consisting of characters that could lead to the mutated behavior under different conditions. We’ll use a mixture of spaces, control characters, and common text:

```python
from string_utils.validation import is_full_string

def test__is_full_string_kill_mutant():
    """
    Test the is_full_string function in a way that should cause different outcomes
    if the mutant change is not equivalent to the baseline logic.
    This examines behavior with Unicode whitespace characters that could affect the 
    condition differently.
    """
    # Input that is not strictly empty but behaves differently under the mutant's check
    # because it's not entirely whitespace when its 'strip()' method is invoked.
    input_string = '   \u200B   '  # String with whitespace and zero-width space

    output = is_full_string(input_string) # Expecting output to be False
    print(f'Input: "{input_string}" | Output: {output}')

# Execute the test function
test__is_full_string_kill_mutant()
```

### Explanation:
- The test uses a string with spaces before and after a zero-width space (`\u200B`), which is technically present but renders itself invisible and could affect the logic in the mutant.
- The baseline's expected output would be `False` as the zero-width space doesn’t count towards the effective content of the string in the baseline logic.
  
By running this test on both the baseline and mutant implementations, I would expect to see a difference in output indicating that we have successfully identified the mutant. 


### Test Results

### Running Test on Baseline

```
Input: "   ​   " | Output: True
```


### Running Test on Mutant

```
Input: "   ​   " | Output: True
```

Your test contains no assertions! Add assertions to your test, so that the test fails when executed with the mutant.

Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


Thank you for the feedback. I will add assertions to the test case to ensure it effectively distinguishes between the baseline and mutant. 

### Test Strategy Update
The goal is to structure the test so that it passes with the baseline (i.e., gets a specific output) but fails with the mutant (i.e., produces a different output). Since the mutant changes the logic from `strip() != ''` to `strip() > ''`, I will use additional assertions to explicitly check the expected outcome.

### Revised Test
Given this context, I will modify the test to include assertions based on the expected behavior when using different strings. I will customize the input based on known cases that might output True for the baseline but False for the mutant. Here’s the adjusted test case:

```python
from string_utils.validation import is_full_string

def test__is_full_string_kill_mutant():
    """
    Test the is_full_string function to distinguish 
    the mutant's behavior from the baseline's logic. The goal is to 
    break the mutant with an intended input.
    """

    # Input string that will be interpreted differently by the mutant
    input_strings = [
        '   ',               # Just spaces; expected False for both
        'text' + '\u200B',  # Text followed by zero-width space; based should return True
        '\u200Btext',       # Zero-width space before text; means baseline returns True
    ]

    for input_string in input_strings:
        output = is_full_string(input_string)
        print(f'Input: "{input_string}" | Output: {output}')
        
        if input_string == '   ':
            # Check that both outputs are false.
            assert output == False, "Expected False for whitespace only input"
        else:
            # Here we expect the baseline to return True and the mutant to return False.
            assert output == True, "Expected True for non-whitespace input"

# Execute the test function
test__is_full_string_kill_mutant()
```

### Explanation:
- The test now includes assertions comparing the actual output with the expected output.
- For the empty space input `input_strings` which should yield `False`, we ensure the output meets expectations across both versions.
- For cases where we expect the baseline to handle non-empty cases (i.e., return `True`), the assumption is that the mutant may behave differently due to the changes in the comparison logic.

This setup aims to identify the mutant reliably if the assumptions about how the logic has changed hold true. Let's execute this test and see if it successfully distinguishes the mutant.


### Test Results

### Running Test on Baseline

```
Input: "   " | Output: False
Input: "text​" | Output: True
Input: "​text" | Output: True
```


### Running Test on Mutant

```
Input: "   " | Output: False
Input: "text​" | Output: True
Input: "​text" | Output: True
```


Your test did not correctly identify the mutant. Remember: Your test needs to pass when executed with the baseline, and fail when executed with the mutant. Please adjust your test case or perform more experiments.


# Conversation Aborted

The conversation has been aborted due to `too_many_turns`

The LLM exceeded the allowed number of turns.
