from dataclasses_json.undefined import _UndefinedParameterAction

def test():
    assert isinstance(_UndefinedParameterAction.__dict__["create_init"], staticmethod)
