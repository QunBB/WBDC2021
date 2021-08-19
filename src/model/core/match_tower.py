import tensorflow as tf
import tensorflow.contrib.slim as slim

from src.model.core.utils import embedding_layer, mlp_layer, inputs_from_hashtable, create_emb_table, copy_mask_feed_data


class MatchTower:

    def __init__(self, user_mlp_size, feed_mlp_size, logits_size, input_dropout, dropout, l2_reg,
                 user_vocab_dict, feed_vocab_dict, tables, hashtables,
                 emb_file=None, emb_dim_dict=None, feed_mask_num=None):
        self.user_mlp_size = user_mlp_size
        self.feed_mlp_size = feed_mlp_size
        self.logits_size = logits_size
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.user_vocab_dict = user_vocab_dict
        self.feed_vocab_dict = feed_vocab_dict
        self.tables = tables
        self.hashtables = hashtables
        self.feed_mask_num = feed_mask_num

        all_vocab_dict = user_vocab_dict.copy()
        all_vocab_dict.update(feed_vocab_dict)

        self.emb_tables, self.feed_weight_table = create_emb_table(vocab_size_dict=all_vocab_dict,
                                                                   emb_file=emb_file,
                                                                   emb_dim_dict=emb_dim_dict,
                                                                   feed_num=len(feed_vocab_dict))

    def __call__(self, inputs, is_training):

        inputs_from_hashtable(inputs, hashtable=self.hashtables, table=self.tables,
                              features_name=self.feed_vocab_dict.keys())

        if is_training and self.feed_mask_num is not None:
            tf.logging.warn("Model will mask feed, mask num is: {}".format(self.feed_mask_num))
            copy_mask_feed_data(inputs, mask_num=self.feed_mask_num)

        emb_list, _ = embedding_layer(self.emb_tables, inputs, max_seq_len_dict={})

        user_inputs = tf.concat([tensor for name, tensor in emb_list.items() if name in self.user_vocab_dict], axis=-1)

        # [batch_size, 1, dim] -> [batch_size, feed_feat_num, dim]
        feed_inputs = tf.concat(
            [tf.expand_dims(tensor, axis=1) for name, tensor in emb_list.items() if name in self.feed_vocab_dict],
            axis=-2)
        # [batch_size, feed_feat_num]
        feed_weight = tf.nn.embedding_lookup(self.feed_weight_table, inputs['feedid'])
        feed_weight = tf.nn.softmax(feed_weight)
        # [batch_size, feed_feat_num, 1]
        feed_weight = tf.expand_dims(feed_weight, axis=-1)

        # [batch_size, dim]
        feed_inputs = tf.reduce_sum(feed_inputs * feed_weight, axis=1)

        user_embedding = mlp_layer(user_inputs, hidden_size=self.user_mlp_size, scope="user_mlp",
                                   is_training=is_training, dropout=self.input_dropout, l2_reg=self.l2_reg)
        feed_embedding = mlp_layer(feed_inputs, hidden_size=self.feed_mlp_size, scope="feed_mlp",
                                   is_training=is_training, dropout=self.input_dropout, l2_reg=self.l2_reg)

        fc_inputs = tf.concat([user_embedding * feed_embedding, user_embedding + feed_embedding], axis=-1)
        logits = mlp_layer(fc_inputs, hidden_size=self.logits_size, scope="fc_layer",
                           is_training=is_training, dropout=self.dropout, l2_reg=self.l2_reg)
        output = tf.layers.dense(logits, 1, kernel_initializer=slim.variance_scaling_initializer())
        output = tf.reshape(output, [-1])
        output = tf.nn.sigmoid(output)

        if 'labels' not in inputs.keys():
            labels = output
            tf.logging.warn("Match Tower has not labels, is_training: {}".format(is_training))
        else:
            labels = tf.cast(inputs['labels'], tf.float32)

        epsilon = 1e-8
        loss = tf.negative(labels * tf.log(output + epsilon) + (
                1 - labels) * tf.log(1 - output + epsilon))

        loss = tf.reduce_mean(loss)

        return loss, output, logits, labels
