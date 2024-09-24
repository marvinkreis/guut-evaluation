from string_utils.manipulation import slugify

def test_slugify():
    # Test for proper slug creation without duplicates
    result = slugify('Top 10 Reasons To Love Dogs!!!')
    expected = 'top-10-reasons-to-love-dogs'
    assert result == expected, f'Expected "{expected}", got "{result}"'

    # Test for handling multiple spaces
    result = slugify('This   is   a  test   string')
    expected = 'this-is-a-test-string'
    assert result == expected, f'Expected "{expected}", got "{result}"'

    # Test for handling special characters and multiple spaces leading to duplicates
    result = slugify('Hello!!!   @World!!!')
    expected = 'hello-world'
    assert result == expected, f'Expected "{expected}", got "{result}"'

    # Test for preserving normal functionality with edge cases
    result = slugify(' Mönstér Mägnët ')
    expected = 'monster-magnet'
    assert result == expected, f'Expected "{expected}", got "{result}"'

    # Test with just separators
    result = slugify('-----')
    expected = ''
    assert result == expected, f'Expected "{expected}", got "{result}"'

    # Additional test to check if the mutant fails here
    result = slugify('one---two---three')
    expected = 'one-two-three'
    assert result == expected, f'Expected "{expected}", got "{result}"'