class Dialogue(object):
    def __init__(self, utterances=None):
        self.utterances = list() if utterances is None else utterances


class Utterances(object):
    def __init__(self, tokenized=None, metadata=None):
        self.tokenized = list() if tokenized is None else tokenized
        self.metadata = dict() if metadata is None else metadata
