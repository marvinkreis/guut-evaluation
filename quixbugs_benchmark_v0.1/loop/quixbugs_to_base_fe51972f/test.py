from to_base import to_base

def test__to_base():
    cases = [
        (31, 16, '1F'),
        (255, 16, 'FF'),
        (100, 8, '144'),
        (64, 2, '1000000'),
        (12345, 36, '9IX')
    ]
    for num, base, expected in cases:
        result = to_base(num, base)
        assert result == expected, f"Expected {expected} but got {result} for num {num} and base {base}"