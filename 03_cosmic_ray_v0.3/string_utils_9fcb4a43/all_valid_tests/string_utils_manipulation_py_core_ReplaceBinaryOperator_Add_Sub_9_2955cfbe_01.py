from string_utils.manipulation import prettify

# Refined test cases with clear expected behaviors.
test_cases = [
    ('   multiple..spaces..between..   ', 'Multiple. Spaces. Between.'),  # Expect to reduce multiple dots
    ('   some excessive punctuation!!!   ', 'Some excessive punctuation!'),  # Should trim and condense
    ('hello   world ... ', 'Hello world.'),  # Should finalize with a single period
    ('before.. after one! ....', 'Before. After one!.'),
    ('  no end punctuation  ', 'No end punctuation.'),  # Should add period
    ('   trailing commas,...   ', 'Trailing commas...'),  # Should retain handling
    ('....multiple dots....', 'Multiple dots.'),  # Handling excessive dots
    ('...nothing at the end...   ', 'Nothing at the end.')  # Simplifying at the ends
]

# Running tests to check output against expectations
for i, (input_str, expected) in enumerate(test_cases, start=1):
    print(f"Test Case {i}:")
    output = prettify(input_str)
    print(f"Input: '{input_str}'")
    print(f"Output: '{output}'")
    print(f"Expected Output: '{expected}'")
    print(f"Output matches Expected: {output == expected}\n")