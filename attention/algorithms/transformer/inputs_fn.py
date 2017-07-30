import tensorflow as tf


def make_example(context, answer):
    ex = tf.train.SequenceExample()

    ex.feature_lists.feature_list["context"].feature.add().int64_list.value.extend(context)
    ex.feature_lists.feature_list["answer"].feature.add().int64_list.value.extend(answer)
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
            last_utterance.tokenized.extend(utterance.tokenized)
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
                features["context"] = features["answer"]
                features["answer"] = None


def create_tf_records(dialogue_gen, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    for features in create_sample(dialogue_gen):
        ex = make_example(**features)
        writer.write(ex.SerializeToString())
    writer.close()


def create_textline_file(dialogue_gen, context_filename, answer_filename):
    with open(context_filename, "w") as context_file, open(answer_filename, "w") as answer_file:
        for features in create_sample(dialogue_gen):
            context_file.write(" ".join([str(x) for x in features["context"]]) + "\n")
            answer_file.write(" ".join([str(x) for x in features["answer"]]) + "\n")


def get_input_fn(batch_size, num_epochs, context_filename, answer_filename, max_sequence_len=50):
    def input_fn():
        source_dataset = tf.contrib.data.TextLineDataset(context_filename)
        target_dataset = tf.contrib.data.TextLineDataset(answer_filename)

        def map_dataset(dataset):
            dataset = dataset.map(lambda string: tf.string_split([string]).values)
            dataset = dataset.map(lambda token: tf.string_to_number(token, tf.int64))
            dataset = dataset.map(lambda token: token[:max_sequence_len])
            dataset = dataset.map(lambda tokens: (tokens, tf.size(tokens)))
            return dataset

        source_dataset = map_dataset(source_dataset)
        target_dataset = map_dataset(target_dataset)

        dataset = tf.contrib.data.Dataset.zip((source_dataset, target_dataset))
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=((tf.TensorShape([max_sequence_len]), tf.TensorShape([])),
                                                      (tf.TensorShape([max_sequence_len]), tf.TensorShape([]))
                                                      ))

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        return next_element, None

    return input_fn


"""
def get_input_fn_dataset(batch_size, num_epochs, writer_filename):

    def input_fn():
        def _parse_function(example_proto):
            sequence_features = {
                "context": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                "answer": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            }
            _, parsed_features = tf.parse_single_sequence_example(serialized=example_proto,
                                                                  sequence_features=sequence_features)
            return parsed_features["answer"], parsed_features["context"]

        dataset = tf.contrib.data.TFRecordDataset([writer_filename])
        dataset.map(lambda x: _parse_function(x))
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.padded_batch(batch_size)
        dataset = dataset.shuffle(buffer_size=10000)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        return next_element

    return input_fn


def get_input_fn_old_way(batch_size, num_epochs, writer_filename):

    def input_fn():
        def _parse_function(example_proto):
            sequence_features = {
                "context": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                "answer": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            }
            _, parsed_features = tf.parse_single_sequence_example(serialized=example_proto,
                                                                  sequence_features=sequence_features)
            return parsed_features

        _, example_proto = tf.TFRecordReader().read(tf.train.string_input_producer([writer_filename]))
        sequence_feature = _parse_function(example_proto)
        sequence_features = {
            "context": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "answer": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }
        _, parsed_features = tf.parse_single_sequence_example(serialized=example_proto,
                                                              sequence_features=sequence_features)

        return tf.train.batch(tensors=sequence_feature,
                              batch_size=batch_size,
                              dynamic_pad=True,
                              allow_smaller_final_batch=False,
                              capacity=100 * batch_size)

    return input_fn
"""
