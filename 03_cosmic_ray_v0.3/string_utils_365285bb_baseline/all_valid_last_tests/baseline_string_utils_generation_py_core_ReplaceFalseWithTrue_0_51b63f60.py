import uuid as uuid_lib

# Original UUID function simulating correct behavior
def original_uuid(as_hex: bool = False) -> str:
    uid = uuid_lib.uuid4()
    if as_hex:
        return uid.hex
    return str(uid)

# Mutant UUID function simulating the mutant behavior
def mutant_uuid(as_hex: bool = True) -> str:
    uid = uuid_lib.uuid4()
    if as_hex:
        return uid.hex
    return str(uid)

def test_uuid():
    # Step 1: Test the original function with no parameters
    original_output_no_param = original_uuid()  # Should be a standard UUID with dashes
    assert len(original_output_no_param) == 36  # Length of standard UUID
    assert '-' in original_output_no_param  # Must contain dashes

    # Step 2: Test the original function with as_hex=True
    original_output_hex = original_uuid(as_hex=True)  # Should be a hex UUID
    assert len(original_output_hex) == 32  # Length should be 32 for hex
    assert '-' not in original_output_hex  # Hex should not have dashes

    # Step 3: Test the mutant function with no parameters
    mutant_output_no_param = mutant_uuid()  # Should return a hex UUID by default
    assert len(mutant_output_no_param) == 32  # Must be length of hex UUID
    assert '-' not in mutant_output_no_param  # Hex representation must not have dashes

    # Step 4: Ensure original and mutant outputs are distinct
    assert original_output_no_param != mutant_output_no_param  # The outputs should be different

    # Step 5: Test the mutant function explicitly as standard UUID
    mutant_standard_output = mutant_uuid(as_hex=False)  # Should yield a standard UUID
    assert len(mutant_standard_output) == 36  # Must be length of standard UUID
    assert '-' in mutant_standard_output  # Should contain dashes

    # Step 6: Detect mutant behavior
    if original_output_no_param.replace('-', '') == mutant_output_no_param:
        raise AssertionError("Mutant detected: The function incorrectly returns a hex format UUID when called without parameters.")

# Run the test
test_uuid()