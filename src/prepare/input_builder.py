import tensorflow as tf

from config.config import *


def file_based_input_fn_builder(input_file, max_seq_length, is_training, batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "device": tf.FixedLenFeature([], tf.int64),
        "read_comment": tf.FixedLenFeature([], tf.int64),
        "comment": tf.FixedLenFeature([], tf.int64),
        "like": tf.FixedLenFeature([], tf.int64),
        "click_avatar": tf.FixedLenFeature([], tf.int64),
        "forward": tf.FixedLenFeature([], tf.int64),
        "follow": tf.FixedLenFeature([], tf.int64),
        "favorite": tf.FixedLenFeature([], tf.int64),
        "history_seq_len": tf.FixedLenFeature([], tf.int64),
        "feedid_origin_history": tf.FixedLenFeature([max_seq_length['feedid']], tf.int64),
        "interval_history": tf.FixedLenFeature([max_seq_length['feedid']], tf.int64),
        "behaviors": tf.FixedLenFeature([], tf.string),
        "feedid_origin": tf.FixedLenFeature([], tf.int64),
        "userid": tf.FixedLenFeature([], tf.int64),

        "userid_origin": tf.FixedLenFeature([], tf.int64),

        "count_features": tf.FixedLenFeature([COUNT_FEATURES_LEN], tf.float32),

    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    d = tf.data.TFRecordDataset(input_file)
    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=10000)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=False))

    iters = d.make_one_shot_iterator()
    batch = iters.get_next()

    # behaviors是多维数组，从bytes恢复
    # TODO 标签数量: 7
    batch['behaviors'] = tf.reshape(tf.decode_raw(batch['behaviors'], tf.float32),
                                    shape=[-1, max_seq_length['feedid'], 7])

    return batch
