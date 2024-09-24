# If the original regex definition was not provided, let's retrieve it
from string_utils._regex import PRETTIFY_RE

def check_regex_definition():
    right_space_regex = PRETTIFY_RE['RIGHT_SPACE']
    print(f"RIGHT_SPACE regex: {right_space_regex.pattern}")  # Example