# Results

The LLM found a test case that detects the mutant.

## Mutant

```diff mutant.diff
{{ problem.mutant_diff() | rtrim }}
```

## Test Case

```python
{{ test | trim }}
```

## Running Test on Baseline

```
{{ result.correct | format_test_result }}
```
{% if result.correct.timeout %}
The test was canceled due to a timeout.
{% endif %}
{% if result.correct.exitcode != 0 %}
The test exited with exitcode {{ result.correct.exitcode }}.
{% endif %}

## Running Test on Mutant

```
{{ result.mutant | format_test_result }}
```
{% if result.mutant.timeout %}
The test was canceled due to a timeout.
{% endif %}
{% if result.mutant.exitcode != 0 %}
The test exited with exit code {{ result.mutant.exitcode }}.
{% endif %}
