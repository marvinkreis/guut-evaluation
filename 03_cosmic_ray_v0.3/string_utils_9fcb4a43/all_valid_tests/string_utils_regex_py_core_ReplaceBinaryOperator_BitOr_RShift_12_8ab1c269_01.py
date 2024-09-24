# Test input
test_string = "Jane's and John's books."

# Compile original regex
try:
    original_regex = PRETTIFY_RE['SAXON_GENITIVE']
    original_match = original_regex.findall(test_string)
    print(f"Original regex compiled and matched: {original_match}")
except Exception as e:
    print(f"Original regex raised an exception: {e}")

# Compile mutant regex
try:
    mutant_regex = mutant_PRETTIFY_RE['SAXON_GENITIVE']
    mutant_match = mutant_regex.findall(test_string)
    print(f"Mutant regex compiled and matched: {mutant_match}")
except Exception as e:
    print(f"Mutant regex raised an exception: {e}")