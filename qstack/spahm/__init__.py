from qstack.tools import FrozenKeysDict


class GuessesDict(FrozenKeysDict):
    def __init__(self, dictionary=None):
        _omod_fns_names = ('core', 'sad', 'sap', 'gwh', 'lb', 'huckel', 'lb-hfs')
        super().__init__(_omod_fns_names, dictionary)
