import tensorflow as tf
import numpy as np
import math
import six

import tensorflow.contrib.slim as slim

from config.config import *


def multi_loss_tradeoff(loss_dict):
    total_loss = 0
    for name in loss_dict.keys():
        loss_weight_log = tf.get_variable("loss_weight_log_"+name, initializer=[0.], dtype=tf.float32, trainable=True)
        # total_loss += tf.exp(-loss_weight_log) * loss_dict[name] + loss_weight_log
        total_loss += tf.exp(loss_weight_log) * loss_dict[name] + tf.exp(-loss_weight_log)
    return total_loss


def copy_mask_feed_data(inputs, mask_num):
    """
    取前mask_num个样本，将feedid置为0。最好保证inputs是乱序的
    :param inputs:
    :param mask_num:
    :return:
    """
    mask_inputs = {name: tensor[:mask_num] for name, tensor in inputs.items()}
    mask_inputs['feedid'] = tf.minimum(mask_inputs['feedid'], 0)
    for name in inputs.keys():
        inputs[name] = tf.concat([inputs[name], mask_inputs[name]], axis=0)


def create_emb_table(vocab_size_dict, emb_file, emb_dim_dict, feed_num=None):
    """
    创建embedding table
    :param vocab_size_dict:
    :param emb_file:
    :param emb_dim_dict:
    :param feed_num:
    :return:
    """
    with tf.variable_scope("share_embedding_table", reuse=tf.AUTO_REUSE):
        emb_tables = {}

        for name in vocab_size_dict.keys():

            if emb_file is not None and emb_file.get(name) is not None:
                feed_emb = np.load(emb_file.get(name))

                emb_tables[name] = tf.get_variable(name + "_emb_table",
                                                   initializer=feed_emb.astype(np.float32),
                                                   dtype=tf.float32)
            else:
                emb_tables[name] = tf.get_variable(name + "_emb_table",
                                                   shape=[vocab_size_dict[name], emb_dim_dict[name]],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                   )
            print("creating embedding table: {}, {}".format(name, emb_tables[name].shape))
        if feed_num is not None:
            feed_weight_table = tf.get_variable("feed_weight_table",
                                                shape=[emb_tables['feedid'].shape.as_list()[0], feed_num],
                                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            return emb_tables, feed_weight_table

    return emb_tables


def inputs_from_hashtable(inputs, hashtable, table, features_name):
    """
    从tensorflow hashtable中补充样本输入
    :param inputs:
    :param hashtable:
    :param table:
    :param features_name:
    :return:
    """
    # test阶段只有userid_origin
    if 'userid' not in inputs.keys():
        for name in ['feedid_origin_history', 'interval_history', 'user_count_features']:
            inputs[name] = tf.nn.embedding_lookup(table[name],
                                                  hashtable['userid_origin_id'].lookup(inputs['userid_origin']))
        inputs['behaviors'] = tf.nn.embedding_lookup(table['his_behaviors'],
                                                     hashtable['userid_origin_id'].lookup(inputs['userid_origin']))
        inputs['userid'] = hashtable['userid'].lookup(inputs['userid_origin'])
        inputs['history_seq_len'] = hashtable['history_seq_len'].lookup(inputs['userid_origin'])

        inputs['feed_count_features'] = tf.nn.embedding_lookup(table['feed_count_features'],
                                                               hashtable['feedid_origin_id'].lookup(
                                                                   inputs['feedid_origin']))
        inputs['count_features'] = tf.concat([inputs['user_count_features'], inputs['feed_count_features']], axis=-1)

    for name in features_name:
        if name in ['tag', 'word', 'keyword']:
            inputs[name] = tf.nn.embedding_lookup(table[name],
                                                  hashtable['feedid_origin_id'].lookup(inputs['feedid_origin']))
            inputs[name + "_len"] = hashtable[name + "_len"].lookup(inputs['feedid_origin'])
        else:
            inputs[name] = hashtable[name].lookup(inputs['feedid_origin'])

    # match模型没有history
    if 'feedid_origin_history' in inputs.keys():
        for name in features_name:
            if name in ['tag', 'word', 'keyword']:
                inputs[name + "_history"] = tf.nn.embedding_lookup(table[name],
                                                                   hashtable['feedid_origin_id'].lookup(
                                                                       inputs['feedid_origin_history']))
                inputs[name + "_seq_len"] = hashtable[name + "_len"].lookup(inputs['feedid_origin_history'])
            else:
                inputs[name + "_history"] = hashtable[name].lookup(inputs['feedid_origin_history'])


def bayes_layer(inputs, is_training, specific_size, dropout, l2_reg):
    """
    贝叶斯loss网络
    :param inputs:
    :param is_training:
    :param specific_size:
    :param dropout:
    :param l2_reg:
    :return:
    """
    outputs = {}
    # 第一层目标
    for name in ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment']:
        outputs[name] = mlp_layer(inputs[name], is_training, specific_size, scope=name,
                                  dropout=dropout, l2_reg=l2_reg)
    # 第二层目标
    bayes_relation = {"follow": "click_avatar"}
    for name in ['follow']:
        outputs[name] = mlp_layer(tf.concat([inputs[name], outputs[bayes_relation[name]]], axis=-1),
                                  is_training, specific_size, scope=name,
                                  dropout=dropout, l2_reg=l2_reg)

    return outputs


def embedding_layer(emb_tables, inputs, max_seq_len_dict):
    """
    embedding lookup操作
    :param emb_tables:
    :param inputs:
    :param max_seq_len_dict:
    :return:
    """
    emb_list = {}
    for name, table in emb_tables.items():
        if name in ['tag', 'word', 'keyword']:
            # [batch_size, len, dim]
            tag_emb = tf.nn.embedding_lookup(table, inputs[name])
            tag_mask = tf.sequence_mask(inputs[name + '_len'], CTR_MAX_LEN[name], dtype=tf.float32)
            emb_list[name] = tf.reduce_sum(tag_emb * tf.expand_dims(tag_mask, axis=-1), axis=1) / tf.reduce_sum(
                tag_mask, axis=-1, keepdims=True)
        elif name == 'interval':
            continue
        else:
            emb_list[name] = tf.nn.embedding_lookup(table, inputs[name])

    his_emb_dict = {}
    for name in max_seq_len_dict.keys():
        # [batch_size, max_seq_len, dim]
        his_emb_dict[name] = tf.nn.embedding_lookup(emb_tables[name], inputs[name + '_history'])

    return emb_list, his_emb_dict


def self_attention_layer(max_seq_len_dict, inputs, his_emb_dict, emb_tables):
    self_attention_dict = {}
    for name in max_seq_len_dict.keys():
        with tf.variable_scope(name):
            mask = tf.sequence_mask(inputs[name + '_seq_len'], max_seq_len_dict[name], dtype=tf.float32)
            attention_mask = create_attention_mask_from_input_mask(
                his_emb_dict[name], mask)
            self_attention = attention_layer(
                from_tensor=his_emb_dict[name],
                to_tensor=his_emb_dict[name],
                attention_mask=attention_mask,
                num_attention_heads=1,
                size_per_head=emb_tables[name].shape.as_list()[-1],
                attention_probs_dropout_prob=0.0,
                initializer_range=0.02,
                do_return_2d_tensor=False,
                batch_size=None,
                from_seq_length=max_seq_len_dict[name],
                to_seq_length=max_seq_len_dict[name])
            self_attention_dict[name] = self_attention

    return self_attention_dict


def history_attention_layer(max_seq_len_dict, inputs, self_attention_dict, emb_dict):
    his_attention_dict = {}
    for name in max_seq_len_dict.keys():
        with tf.variable_scope(name):
            mask = tf.sequence_mask(inputs[name + '_seq_len'], max_seq_len_dict[name], dtype=tf.float32)
            his_attention = _attention(self_attention_dict[name], emb_dict[name], mask)
        his_attention_dict[name] = his_attention
    return his_attention_dict


# def _attention(k, v, mask):
#     k_weight = tf.matmul(k, tf.expand_dims(v, axis=-1))
#
#     sum_pooling = tf.reduce_sum(k * k_weight * tf.expand_dims(mask, axis=-1), axis=1)
#
#     return sum_pooling


def _attention(k, v, mask, is_training, dropout=0., hidden_size=[80, 40], l2_reg=0.):
    """

    :param k: [batch_size, seq_len, dim]
    :param v: [batch_size, dim]
    :param mask: [batch_size, seq_len]
    :return:
    """
    seq_len = k.shape.as_list()[1]
    # [batch_size, seq_len, dim]
    v = tf.tile(tf.expand_dims(v, axis=1), [1, seq_len, 1])
    # [batch_size, seq_len, dim * 4]
    inputs = tf.concat([k, v, k - v, k * v], axis=-1)

    k_weight = None
    for size in hidden_size:
        k_weight = dnn_layer(inputs, size, is_training, dropout, l2_reg)
    k_weight = tf.layers.dense(k_weight, 1)

    sum_pooling = tf.reduce_sum(k * k_weight * tf.expand_dims(mask, axis=-1), axis=1)

    return sum_pooling


def dnn_dropout(inputs, is_training, dropout):
    if is_training:
        return tf.nn.dropout(inputs, 1 - dropout)

    return inputs


def mlp_layer(inputs, is_training, hidden_size, scope, dropout, l2_reg):
    with tf.variable_scope(scope):
        specific_dnn = inputs
        for size in hidden_size:
            specific_dnn = dnn_layer(specific_dnn, size, is_training, dropout, l2_reg)

    return specific_dnn


def compute_loss(logits, label, scope, negative_decay, l2_reg):
    logits = tf.layers.dense(logits, 1,
                             kernel_initializer=slim.variance_scaling_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    epsilon = 1e-8
    logits = tf.reshape(logits, [-1])
    prob = tf.nn.sigmoid(logits)

    label = tf.cast(label, tf.float32)
    # 对于负样本，减弱loss的作用
    # 对于正样本，则保持loss不变
    loss = label * tf.log(prob + epsilon) + negative_decay[scope] * (1 - label) * tf.log(
        1 - prob + epsilon)
    loss = tf.negative(loss)

    loss = tf.reduce_mean(loss)

    return loss, prob


def dnn_layer(inputs, hidden_size, is_training, dropout, l2_reg):
    output = tf.layers.dense(inputs, hidden_size,
                             kernel_initializer=slim.variance_scaling_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    output = tf.layers.batch_normalization(output, training=is_training)
    # output = tf.contrib.layers.layer_norm(
    #     inputs=output, begin_norm_axis=-1, begin_params_axis=-1, center=False, scale=False)
    output = gelu(output)
    output = dnn_dropout(output, is_training, dropout)

    return output


def cross_layer(inputs, l2_reg, scope, hidden_size=None):
    if hidden_size is None:
        hidden_size = inputs.shape.as_list()[-1] // 2

    output = tf.layers.dense(inputs, hidden_size,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                             name=scope + "_1")
    output = gelu(output)

    output = tf.layers.dense(output, inputs.shape.as_list()[-1],
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                             name=scope + "_2")

    return gelu(output + inputs)


def external_attention(inputs, hidden_size, mask, l2_reg):
    """
    :param mask: [batch_size, seq_len]
    :param l2_reg:
    :param hidden_size:
    :param inputs: [batch_size, seq_len, dim]
    :return:
    """
    attention = tf.layers.dense(inputs, hidden_size,
                                kernel_initializer=slim.variance_scaling_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                name='memory_k')
    mask = tf.expand_dims(mask, axis=-1)
    mask = (1.0 - tf.cast(mask, tf.float32)) * -10000.0
    attention = attention + mask
    epsilon = 1e-12
    attention = tf.nn.softmax(attention, axis=1)
    attention = tf.divide(attention,
                          tf.maximum(tf.reduce_sum(attention, axis=-1, keepdims=True), epsilon))

    output = tf.layers.dense(attention, inputs.shape.as_list()[-1],
                             kernel_initializer=slim.variance_scaling_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                             name='memory_v')
    return output + inputs


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def dice(x, training):
    _x = tf.layers.batch_normalization(x, center=False, scale=False, training=training)
    px = tf.sigmoid(_x)
    alphas = tf.get_variable('alpha_' + "".join(x.name.split(":")), _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    return alphas * (1 - px) * _x + px * _x


def history_emb_pooling(his_emb_dict, inputs):
    for name in ['word', 'tag', 'keyword']:
        if his_emb_dict.get(name) is None:
            continue
        # [batch_size, his_len, f_len, dim]
        his_emb = his_emb_dict[name]
        # [batch_size, his_len, f_len]
        mask = tf.sequence_mask(inputs[name + '_seq_len'], CTR_MAX_LEN[name], dtype=tf.float32)
        his_emb_dict[name] = tf.reduce_sum(his_emb * tf.expand_dims(mask, axis=-1), axis=-2) / tf.reduce_sum(
            mask, axis=-1, keepdims=True)


############### Transformer ################
def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.

    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=layer_input,
                        to_tensor=layer_input,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                # Run a linear projection of `hidden_size` then add a residual
                # with `layer_input`.
                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                    attention_output = dropout(attention_output, hidden_dropout_prob)
                    attention_output = layer_norm(attention_output + layer_input)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range))

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      size_per_head: int. Size of each attention head.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      initializer_range: float. Range of the weight initializer.
      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].
      batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
      from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
      to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.

    Returns:
      float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*H]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
