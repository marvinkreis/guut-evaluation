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
index 780de34..9057229 100644
--- a/string_utils/manipulation.py
+++ b/mutant/string_utils/manipulation.py
@@ -218,11 +218,11 @@ class __StringFormatter:
 
     def __uppercase_first_char(self, regex_match):
         return regex_match.group(0).upper()
 
     def __remove_duplicates(self, regex_match):
-        return regex_match.group(1)[0]
+        return regex_match.group( 2)[0]
 
     def __uppercase_first_letter_after_sign(self, regex_match):
         match = regex_match.group(1)
         return match[:-1] + match[2].upper()
```

# Debugging


## Hypothesis
Given that the mutant modifies the `__remove_duplicates` method in the `__StringFormatter` class to use `regex_match.group(2)` instead of `regex_match.group(1)`, I hypothesize that when providing input that would result in a match for a regex pattern with at least two capturing groups, the behavior of the mutant will differ from the baseline. Specifically, when a string has duplicate consecutive characters matched by the regex, the mutant will return an unexpected character from the second group of the match instead of the expected first group character. I predict that an input string with consecutive duplicates, such as 'aabb', will lead to different outputs for the baseline and the mutant when passed through the `prettify` function.

## Experiment
```python
from string_utils.manipulation import prettify

def test__string_formatter_duplicate_removal():
    """
    Test the prettify function with input that has consecutive duplicates.
    This will check if the mutant behaves differently from the baseline 
    by checking if the output handles the removal of duplicates correctly.
    """
    input_string = 'aabbcc'
    expected_output = 'abc'  # Expect the duplicates to be removed.

    output = prettify(input_string)
    print(f"output: {output}")
    assert output == expected_output
```


### Experiment Results

### Running Experiment on Baseline
```
output: Aabbcc
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal()
  File "test.py", line 14, in test__string_formatter_duplicate_removal
    assert output == expected_output
AssertionError
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
output: Aabbcc
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal()
  File "test.py", line 14, in test__string_formatter_duplicate_removal
    assert output == expected_output
AssertionError
```
The experiment exited with exit code 1.


## Conclusion
Both the baseline and the mutant produced the same output: `Aabbcc`. This means that the change in the mutant, using `regex_match.group(2)` instead of `regex_match.group(1)` in `__remove_duplicates`, did not alter the output behavior for the input provided. Consequently, this specific test did not successfully distinguish between the baseline and the mutant.

Next, I will adjust my hypothesis to test a different scenario that may trigger the difference in behavior, particularly focusing on the conditions that the modified regex may encounter.

## Hypothesis
Given the prior observations, I hypothesize that since the current test input did not trigger a change, I should use an input string that conforms to the expected regex pattern but has different groups. Specifically, I will test with a regex pattern for duplicate characters but also ensure that the second capturing group is valid. I predict that an input where duplicates are preceded by different characters but matched will reveal different outputs for the baseline and the mutant.

## Experiment
```python
from string_utils.manipulation import prettify

def test__string_formatter_duplicate_removal_new_input():
    """
    Test the prettify function with input that has consecutive duplicates 
    preceded by other characters. We expect different behavior if the 
    mutant's change has an effect.
    """
    input_string = 'xxyyzz'
    expected_baseline_output = 'xyz'  # Expect the duplicates to be removed.

    output = prettify(input_string)
    print(f"output: {output}")
    assert output == expected_baseline_output
```


### Experiment Results

### Running Experiment on Baseline
```
output: Xxyyzz
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal_new_input()
  File "test.py", line 14, in test__string_formatter_duplicate_removal_new_input
    assert output == expected_baseline_output
AssertionError
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
output: Xxyyzz
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal_new_input()
  File "test.py", line 14, in test__string_formatter_duplicate_removal_new_input
    assert output == expected_baseline_output
AssertionError
```
The experiment exited with exit code 1.


## Conclusion
Both the baseline and the mutant produced the same output: `Xxyyzz`, which indicates that the change made in the mutant's `__remove_duplicates` method did not affect the output for the provided input. This means that the mutant has not been detected yet.

Given that both previous inputs did not yield different outputs, I will adjust the hypothesis again. This time, I'll test a string with a different structure that should interact with the regex patterns in a way that could potentially reveal differences between the baseline and the mutant.

## Hypothesis
I hypothesize that an input containing a pattern that leads to a match in the regex for duplicates but in a more complex arrangement could reveal the differences. I will test a multi-character string with specific sequences that allow for duplicates and ensure that the output relies heavily on the capturing groups. I predict that an input like `abba` will lead to an output being affected by how the mutant's `__remove_duplicates` is implemented.

## Experiment
```python
from string_utils.manipulation import prettify

def test__string_formatter_duplicate_removal_complex_input():
    """
    Test the prettify function with complex input involving adjacent duplicates. 
    This aims to validate if the mutant behaves differently as a result of the change made.
    """
    input_string = 'abbaa'  # The expected result should be 'aba'
    expected_output_baseline = 'aba'  # We expect the duplicates to be seen correctly.

    output = prettify(input_string)
    print(f"output: {output}")
    assert output == expected_output_baseline
```


### Experiment Results

### Running Experiment on Baseline
```
output: Abbaa
Traceback (most recent call last):
  File "test.py", line 14, in <module>
    test__string_formatter_duplicate_removal_complex_input()
  File "test.py", line 13, in test__string_formatter_duplicate_removal_complex_input
    assert output == expected_output_baseline
AssertionError
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
output: Abbaa
Traceback (most recent call last):
  File "test.py", line 14, in <module>
    test__string_formatter_duplicate_removal_complex_input()
  File "test.py", line 13, in test__string_formatter_duplicate_removal_complex_input
    assert output == expected_output_baseline
AssertionError
```
The experiment exited with exit code 1.


## Conclusion
Both the baseline and the mutant produced the same output: `Abbaa`. This means that the change in the mutant, specifically using `regex_match.group(2)` in `__remove_duplicates`, did not alter the output for the provided input when processed through the `prettify` function.

Since the expected output always matches for both the baseline and the mutant and my input strings did not successfully distinguish between them, I will assume that the regex used does not create an opportunity for the differences to manifest with the currently tested inputs and methods.

Next, I will refine my hypothesis and test with input that might produce different outputs based on the `__remove_duplicates` function's behavior. I will specifically craft input that ensures both capturing groups are utilized in a way that can lead to a varying output.

## Hypothesis
I hypothesize that using an input string that results in a regex match where the second capturing group has specific content can yield differing results. If the mutant's implementation where `regex_match.group(2)` is invoked remains untriggered by direct duplicates alone, using mixed character contexts should highlight discrepancies. I predict that an input like `'abccba'` should allow us to observe how the output is affected between the baseline and the mutant.

## Experiment
```python
from string_utils.manipulation import prettify

def test__string_formatter_duplicate_removal_edge_case():
    """
    Test the prettify function with a case involving edge characters and duplicates.
    We expect different behavior if the mutant's change is invoked by the regex behavior.
    """
    input_string = 'abccba'  # Expecting output to be 'abc' after removing adjacent duplicates.

    expected_output_baseline = 'abc'  # The expected result of duplicate removal in baseline.

    output = prettify(input_string)
    print(f"output: {output}")
    assert output == expected_output_baseline
```


### Experiment Results

### Running Experiment on Baseline
```
output: Abccba
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal_edge_case()
  File "test.py", line 14, in test__string_formatter_duplicate_removal_edge_case
    assert output == expected_output_baseline
AssertionError
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
output: Abccba
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal_edge_case()
  File "test.py", line 14, in test__string_formatter_duplicate_removal_edge_case
    assert output == expected_output_baseline
AssertionError
```
The experiment exited with exit code 1.


## Conclusion
Both the baseline and the mutant produced the same output: `Abccba`, indicating that the mutant has not been detected yet. This outcome suggests that the change in the `__remove_duplicates` method of the mutant may not lead to a noticeable difference in this specific context.

Given that various attempts to distinguish the mutant from the baseline using different inputs failed to produce differing outputs, I will take another approach. 

I will attempt to directly observe and manipulate the behavior of the mutated `__remove_duplicates` method by utilizing a string that has different characters and patterns, focusing on a string that would ordinarily be modified by the removal of duplicates in a unique way.

## Hypothesis
I hypothesize that testing a sequence with discrepancies in patterns—where to intervene within a common set of characters will show a deviation in handling when processed through the mutated function. An input like `'abbab'` should force the function to behave differently, depending on how the capturing groups are processed. I predict that both the baseline and the mutant will handle this input in a way that will present different outputs.

## Experiment
```python
from string_utils.manipulation import prettify

def test__string_formatter_duplicate_removal_interference():
    """
    Test the prettify function with a sequence that has specific duplicates and varying characters.
    This test aims to signal a difference via how the mutant handles duplicates as opposed to the baseline.
    """
    input_string = 'abbab'  # Expected to yield 'abab' after processing to remove duplicates effectively.

    expected_output_baseline = 'abab'  # This is what we expect from the baseline behavior.

    output = prettify(input_string)
    print(f"output: {output}")
    assert output == expected_output_baseline
```


### Experiment Results

### Running Experiment on Baseline
```
output: Abbab
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal_interference()
  File "test.py", line 14, in test__string_formatter_duplicate_removal_interference
    assert output == expected_output_baseline
AssertionError
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
output: Abbab
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal_interference()
  File "test.py", line 14, in test__string_formatter_duplicate_removal_interference
    assert output == expected_output_baseline
AssertionError
```
The experiment exited with exit code 1.


## Conclusion
Both the baseline and the mutant produced the same output: `Abbab`, indicating that my latest hypothesis and test case did not succeed in distinguishing between the mutant and the baseline behavior. Despite several different approaches and inputs, the mutant still behaved equivalently to the baseline under the tested conditions.

At this point, it appears that the change introduced to the `__remove_duplicates` method (i.e., switching from `regex_match.group(1)` to `regex_match.group(2)`) does not lead to any observable differences for the range of inputs attempted. This could mean the mutant is equivalent in practice, as it does not affect the outputs of the string manipulation performed by the `prettify` function.

## Equivalent Mutant
I believe the mutant to be equivalent. The alteration from `regex_match.group(1)` to `regex_match.group(2)` in the `__remove_duplicates` function does not appear to have any significant functional effect due to the nature of the regex patterns being used. Based on the numerous input tests conducted, the outputs remained fundamentally unchanged, suggesting that the second capturing group is not utilized in a way to alter the removal of duplicates effectively.

Given that my attempts with various input strings (including many scenarios where duplicate characters could appear) constantly returned identical outputs for both the baseline and the mutant during string manipulation, I conclude the mutant's implementation is effectively equivalent in behavior to the original code.


You claimed that the mutant as equivalent.

Now, try other experiments to check if you can prove yourself wrong. Try new approaches or new inputs you haven't tried yet. See if you can maybe find a difference between the **Baseline** and the **Mutant** anyways.


## Hypothesis
Given that the previous tests did not successfully differentiate the mutant from the baseline, I will explore a different approach that focuses on using multicharacter sequences and varying patterns that might interact differently due to the change in capturing groups. Specifically, I’ll use a string that has complex overlapping patterns. I hypothesize that using an input string that specifically leverages the potential of both capturing groups could show differing behaviors.

I predict that an input string such as `'abcabc'`, where characters are repeated but interspersed in a way that allows for distinct regex matches, could expose differences in output between the baseline and the mutant.

## Experiment
```python
from string_utils.manipulation import prettify

def test__string_formatter_duplicate_removal_repeated_patterns():
    """
    Test the prettify function with an input string featuring repeated patterns.
    This may reveal differences based on how the mutant handles capturing groups in the regex.
    """
    input_string = 'abcabc'  # We expect the return to reflect the removal of duplicates, yielding 'abc'.

    expected_output_baseline = 'abc'  # Expecting duplicates to be effectively removed in baseline.

    output = prettify(input_string)
    print(f"output: {output}")
    assert output == expected_output_baseline
```

### Input Explanation
The input string `'abcabc'` contains a full set of unique characters followed by the same set, which gives the regex a chance to process variations in groups. This structure should highlight any changes made by the mutant's switch to `group(2)`, depending on how the regex logic interprets the characters. 

## Attempt to Run this Experiment


### Experiment Results

### Running Experiment on Baseline
```
output: Abcabc
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal_repeated_patterns()
  File "test.py", line 14, in test__string_formatter_duplicate_removal_repeated_patterns
    assert output == expected_output_baseline
AssertionError
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
output: Abcabc
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal_repeated_patterns()
  File "test.py", line 14, in test__string_formatter_duplicate_removal_repeated_patterns
    assert output == expected_output_baseline
AssertionError
```
The experiment exited with exit code 1.


## Conclusion
Both the baseline and the mutant produced the same output: `Abcabc`. This reinforces the previous conclusion that my hypotheses and experiments have not successfully differentiated between the mutant and the baseline so far.

Since my previous attempts have revolved around variations of character duplications without success, I will now change my approach by exploring input strings that contain special characters or whitespace that may affect how the `prettify` function behaves. 

## Hypothesis
Using strings that include special characters, such as punctuation or whitespace, might trigger unique regex behavior in the `prettify` function that could expose the differences created by the mutant's changes. I predict that an input string formatted as `'Hello!! Hello??'` will likely yield different outputs due to the special character handling in the prettifying process, which might be impacted by the change in the capturing groups.

## Experiment
```python
from string_utils.manipulation import prettify

def test__string_formatter_duplicate_removal_special_chars():
    """
    Test the prettify function with an input string that includes special characters.
    This is to observe if the mutant behaves differently due to changes in regex evaluation.
    """
    input_string = 'Hello!! Hello??'  # We expect the duplicate "Hello" to remain but punctuation to be handled.

    expected_output_baseline = 'Hello! Hello?'  # Expect to keep a single instance of Hello while cleaning punctuation.

    output = prettify(input_string)
    print(f"output: {output}")
    assert output == expected_output_baseline
```

### Input Explanation
The input string `'Hello!! Hello??'` contains multiple punctuations which will likely be formatted or cleaned up by the `prettify()` function. The distinctiveness of the inputs based on grouping might expose differences in functionality between the baseline and mutant implementations. 

## Attempt to Run this Experiment


### Experiment Results

### Running Experiment on Baseline
```
output: Hello!! Hello??
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal_special_chars()
  File "test.py", line 14, in test__string_formatter_duplicate_removal_special_chars
    assert output == expected_output_baseline
AssertionError
```
The experiment exited with exit code 1.

### Running Experiment on Mutant
```
output: Hello!! Hello??
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal_special_chars()
  File "test.py", line 14, in test__string_formatter_duplicate_removal_special_chars
    assert output == expected_output_baseline
AssertionError
```
The experiment exited with exit code 1.


## Conclusion
Both the baseline and the mutant produced the same output: `Hello!! Hello??`. This result once again indicates that my input did not reveal any differences in behavior between the mutant and the baseline. 

Considering the persistent equivalence seen so far with various character patterns and separators, it may be insightful to focus specifically on variations in input complexity that involve various types of whitespace, as previous tests haven't sufficiently engaged these kinds of inputs.

## New Hypothesis
I hypothesize that using input strings with irregular spacing and leading/trailing whitespace might expose the difference in handling by the baseline and the mutant. An input such as `'   Hello    World!   '` could lead to different handling regarding how whitespace is formatted and how duplicates, if any, might be processed differently.

## Experiment
```python
from string_utils.manipulation import prettify

def test__string_formatter_duplicate_removal_whitespace():
    """
    Test the prettify function on a string with irregular whitespaces.
    This is to determine if different whitespace handling reveals differences between the mutant and baseline.
    """
    input_string = '   Hello    World!   '  # Expecting whitespace to be normalized and the output properly formatted.
    
    expected_output_baseline = 'Hello World!'  # Extra spaces should be removed, single space maintained.

    output = prettify(input_string)
    print(f"output: {output}")
    assert output == expected_output_baseline
```
### Input Explanation
The input string contains excessive leading, internal, and trailing whitespace that should be cleaned up by the `prettify()` function. Depending on how the regex captures groups, this may yield different outputs in the baseline and mutant implementations.

## Attempt to Run this Experiment


### Experiment Results

### Running Experiment on Baseline
```
output: Hello World!
```

### Running Experiment on Mutant
```
Traceback (most recent call last):
  File "test.py", line 15, in <module>
    test__string_formatter_duplicate_removal_whitespace()
  File "test.py", line 12, in test__string_formatter_duplicate_removal_whitespace
    output = prettify(input_string)
  File "string_utils/manipulation.py", line 429, in prettify
    formatted = __StringFormatter(input_string).format()
  File "string_utils/manipulation.py", line 264, in format
    out = PRETTIFY_RE['DUPLICATES'].sub(self.__remove_duplicates, out)
  File "string_utils/manipulation.py", line 223, in __remove_duplicates
    return regex_match.group( 2)[0]
IndexError: no such group
```
The experiment exited with exit code 1.

Your experiment resulted in exitcode 0 for the **Baseline** and exitcode 1 for the **Mutant**. This means that your experiment can successfully kill the mutant. Next, you should create a test from your experiment.


## Test Instructions

You have 2 experiments/tests left.

Please continue by writing a test that kills the mutant. Remember: the test should pass when executed on the baseline but fail on the mutant.

## Test


```python
from string_utils.manipulation import prettify

def test__string_formatter_duplicate_removal_whitespace_kills_mutant():
    """
    Test the prettify function on a string with irregular whitespaces.
    This test is designed to kill the mutant by producing an IndexError in the mutant code
    while the baseline will handle it correctly and return a formatted string. 
    The input contains excessive leading and trailing whitespace which should be normalized.
    """
    input_string = '   Hello    World!   '  # This input contains irregular spaces.
    
    expected_output_baseline = 'Hello World!'  # The expected output from the baseline.

    output = prettify(input_string)  # This should execute without errors on baseline.
    print(f"output: {output}")
     
    assert output == expected_output_baseline  # This assertion will pass with the baseline.
```

### Explanation:
This test aims to capture the normal handling of extra spaces in a string by the baseline. Given that the mutant's change causes an `IndexError` due to the reliance on `Group(2)` which may not exist in certain regex matches, this test will pass with the baseline and fail with the mutant, effectively killing it. 

Let’s see the result when we run this test case on both the baseline and the mutant.


# Results

The LLM found a test case that detects the mutant.

## Mutant

```diff mutant.diff
diff --git a/string_utils/manipulation.py b/mutant/string_utils/manipulation.py
index 780de34..9057229 100644
--- a/string_utils/manipulation.py
+++ b/mutant/string_utils/manipulation.py
@@ -218,11 +218,11 @@ class __StringFormatter:
 
     def __uppercase_first_char(self, regex_match):
         return regex_match.group(0).upper()
 
     def __remove_duplicates(self, regex_match):
-        return regex_match.group(1)[0]
+        return regex_match.group( 2)[0]
 
     def __uppercase_first_letter_after_sign(self, regex_match):
         match = regex_match.group(1)
         return match[:-1] + match[2].upper()
```

## Test Case

```python
from string_utils.manipulation import prettify

def test__string_formatter_duplicate_removal_whitespace_kills_mutant():
    """
    Test the prettify function on a string with irregular whitespaces.
    This test is designed to kill the mutant by producing an IndexError in the mutant code
    while the baseline will handle it correctly and return a formatted string. 
    The input contains excessive leading and trailing whitespace which should be normalized.
    """
    input_string = '   Hello    World!   '  # This input contains irregular spaces.
    
    expected_output_baseline = 'Hello World!'  # The expected output from the baseline.

    output = prettify(input_string)  # This should execute without errors on baseline.
    print(f"output: {output}")
     
    assert output == expected_output_baseline  # This assertion will pass with the baseline.
```

## Running Test on Baseline

```
output: Hello World!
```

## Running Test on Mutant

```
Traceback (most recent call last):
  File "test.py", line 18, in <module>
    test__string_formatter_duplicate_removal_whitespace_kills_mutant()
  File "test.py", line 14, in test__string_formatter_duplicate_removal_whitespace_kills_mutant
    output = prettify(input_string)  # This should execute without errors on baseline.
  File "string_utils/manipulation.py", line 429, in prettify
    formatted = __StringFormatter(input_string).format()
  File "string_utils/manipulation.py", line 264, in format
    out = PRETTIFY_RE['DUPLICATES'].sub(self.__remove_duplicates, out)
  File "string_utils/manipulation.py", line 223, in __remove_duplicates
    return regex_match.group( 2)[0]
IndexError: no such group
```
The test exited with exit code 1.
