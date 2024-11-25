from dataclasses_json.undefined import _UndefinedParameterAction

def test():
    assert isinstance(_UndefinedParameterAction.__dict__["handle_to_dict"], staticmethod)
