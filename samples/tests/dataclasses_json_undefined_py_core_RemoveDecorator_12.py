from dataclasses_json.undefined import _CatchAllUndefinedParameters

def test():
    assert isinstance(_CatchAllUndefinedParameters.__dict__["handle_to_dict"], staticmethod)
