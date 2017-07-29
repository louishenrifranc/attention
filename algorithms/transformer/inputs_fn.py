import numpy as np
import tensorflow as tf


def make_example(context, answer):
    ex = tf.train.SequenceExample()
    fl_answer = ex.feature_lists.feature_list["labels"]
    fl_context = ex.feature_lists.feature_list["inputs"]
    for token, in context:
        fl_context.feature.add().int64_list.value.append(token)
    for token, in answer:
        fl_answer.feature.add().int64_list.value.append(token)
    return ex


def filter_and_modify_dialogue(dialogue):
    # Remove single role dialogues
    if len(set([utterance.metadata["role"] for utterance in dialogue.utterances])) < 2:
        return None

    new_dialogue = type(dialogue)()
    last_role = None
    for utterance in dialogue.utterances:
        new_role = utterance.metadata["role"]
        if last_role != new_role:
            if last_role is not None:
                new_dialogue.utterances.append(last_utterance)
            last_role = new_role
            last_utterance = utterance
        else:
            last_utterance.tokenized.append(utterance.tokenized)
    new_dialogue.utterances.append(last_utterance)
    return new_dialogue


def create_sample(dialogue_gen):
    for dialogue in dialogue_gen:
        dialogue = filter_and_modify_dialogue(dialogue)
        if dialogue is None:
            continue
        features = {"context": None, "answer": None}

        for utterance in dialogue.utterances:
            if features["context"] is None:
                features["context"] = utterance.tokenized
            else:
                features["answer"] = utterance.tokenized
                yield features
                features = {"context": None, "answer": None}


def create_tf_records(dialogue_gen, filename):
    with filename as fp:
        writer = tf.python_io.TFRecordWriter(fp.name)
        for features in create_sample(dialogue_gen):
            ex = make_example(**features)
            writer.write(ex.SerializeToString())
        writer.close()


def get_input_fn(batch_size, num_epochs, filename="*.tfrecords"):
    def input_fn():
        features = tf.contrib.data.read_batch_features(file_pattern=tf.gfile.Glob(filename),
                                                       batch_size=batch_size,
                                                       features={
                                                           "context": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                                                           "answer": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                                                       },
                                                       reader=tf.TFRecordReader,
                                                       randomize_input=True,
                                                       num_epochs=num_epochs
                                                       )
        return features, None

    return input_fn
