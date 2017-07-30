import numpy as np
from utils.dialogue import Dialogue, Utterances


def create_mock_dialogue(num_utterances, max_len_tokenized, vocab_size):
    dialogue = Dialogue()
    for _ in range(num_utterances):
        len_utterances = np.random.randint(2, max_len_tokenized, 1)
        tokenized = np.random.randint(1, vocab_size, len_utterances).tolist()
        dialogue.utterances.append(Utterances(tokenized, {"role": np.random.choice(["user", "operator"])}))

    return dialogue


def mock_dialogue_gen(num_samples=100):
    for _ in range(num_samples):
        yield create_mock_dialogue(num_utterances=np.random.randint(2, 15),
                                   max_len_tokenized=np.random.randint(3, 30),
                                   vocab_size=20)
