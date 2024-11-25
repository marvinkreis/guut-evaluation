from httpie.output.streams import RawStream


def test():
    assert RawStream.CHUNK_SIZE_BY_LINE == 1
