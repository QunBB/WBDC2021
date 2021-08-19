import tensorflow as tf
import tensorflow.contrib.slim as slim
from functools import reduce

from src.model.albert.modeling import transformer_model
from src.model.core.fusion_se import SE
from src.model.core.utils import embedding_layer, mlp_layer, history_emb_pooling, \
    compute_loss, bayes_layer, _attention, inputs_from_hashtable, create_emb_table, gelu, multi_loss_tradeoff


class PLE:

    def __init__(self, max_seq_len_dict, sharing_size, specific_layer_num, specific_size, dropout, negative_decay,
                 expert_size, loss_weight, tables, hashtables,
                 input_dropout=0.1, l2_reg=0.0001, behaviors_num=7,
                 emb_file=None, vocab_size_dict=None, emb_dim_dict=None,
                 use_transformer=False, use_transformer_logits=False, use_linear_logits=False, logits_with_label=False):
        self.max_seq_len_dict = max_seq_len_dict
        self.behaviors_num = behaviors_num
        self.sharing_size = sharing_size
        self.specific_size = specific_size
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.negative_decay = negative_decay
        self.loss_weight = loss_weight
        self.input_dropout = input_dropout
        self.specific_layer_num = specific_layer_num
        self.expert_size = expert_size
        self.tables = tables
        self.hashtables = hashtables
        self.use_transformer = use_transformer
        self.use_transformer_logits = use_transformer_logits
        self.use_linear_logits = use_linear_logits
        self.logits_with_label = logits_with_label

        self.feed_features_list = list(vocab_size_dict.keys())
        self.feed_features_list.remove("userid")
        self.feed_features_list.remove("device")

        if use_transformer:
            self.feed_features_list.remove("interval")
            self.interval_table = tf.get_variable("interval_emb_table",
                                                  shape=[vocab_size_dict["interval"], emb_dim_dict["interval"]],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
            vocab_size_dict.pop("interval")

        print("Length of feed features {}: {}".format(len(self.feed_features_list), self.feed_features_list))

        self.emb_tables, self.feed_weight_table = create_emb_table(vocab_size_dict=vocab_size_dict,
                                                                   emb_file=emb_file,
                                                                   emb_dim_dict=emb_dim_dict,
                                                                   feed_num=len(self.feed_features_list))
        self.fusion_se = SE(drop_rate=self.input_dropout,
                            hidden1_size=1536,
                            gating_reduction=8,
                            gating_last_bn=False)

    def __call__(self, inputs, is_training, match_logits=None):
        if match_logits is None:
            # Match tower已经从hashtable加载数据了，这里不应该重复读取
            inputs_from_hashtable(inputs, hashtable=self.hashtables, table=self.tables,
                                  features_name=self.max_seq_len_dict.keys())

        with tf.variable_scope("embedding_layer"):

            # emb_list: [ [batch_size, dim] .....]
            # his_emb_dict: [ [batch_size, max_seq_len, dim] ....]
            emb_list, his_emb_dict = embedding_layer(self.emb_tables, inputs, self.max_seq_len_dict)

            # 历史feed序列对tag、word等进行pooling
            history_emb_pooling(his_emb_dict, inputs)

            ctr_feed = self._features_fusion(emb_list, inputs)
            his_feed = self._features_fusion(his_emb_dict, inputs)

        with tf.variable_scope("attention_layer"):
            mask = tf.sequence_mask(inputs['history_seq_len'], self.max_seq_len_dict['feedid'], dtype=tf.float32)
            # [batch_size, dim]
            his_attention = _attention(his_feed, ctr_feed, mask, is_training=is_training)

        dnn_input = tf.concat([ctr_feed, his_attention, emb_list['userid'], emb_list['device'],
                               inputs['count_features']], axis=-1)

        if self.use_transformer:
            with tf.variable_scope("transformer_layer"):
                transformer_logits, transformer_his = self._transformer_layer(his_feed=his_feed,
                                                                              ctr_feed=ctr_feed,
                                                                              history_seq_len=inputs['history_seq_len'] + 1,
                                                                              max_seq_len=self.max_seq_len_dict['feedid'] + 1,
                                                                              his_interval=inputs['interval_history'],
                                                                              interval_table=self.interval_table)
                dnn_input = tf.concat([dnn_input,
                                       self._history_add_behaviors(transformer_his, inputs['behaviors'], mask)],
                                      axis=-1)

        dnn_input = tf.layers.batch_normalization(dnn_input, training=is_training)

        sharing_dnn = self.fusion_se(dnn_input, is_training=is_training)

        # if is_training:
        #     dnn_input = tf.nn.dropout(dnn_input, 1 - self.input_dropout)
        #
        # sharing_dnn = dnn_input

        with tf.variable_scope("expert_layer"):
            # 共享的expert
            shared_expert_dnn = []
            for i in range(self.specific_layer_num):
                shared_expert_dnn.append(
                    mlp_layer(sharing_dnn, is_training, [self.expert_size], scope="shared_{}".format(i),
                              dropout=self.input_dropout, l2_reg=self.l2_reg))
            # 每个任务独立的expert
            expert_dnn = {}
            for name in self.loss_weight.keys():
                expert_dnn[name] = []
                for i in range(self.specific_layer_num):
                    expert_dnn[name].append(mlp_layer(sharing_dnn, is_training, [self.expert_size],
                                                      scope=name + "_{}".format(i),
                                                      dropout=self.input_dropout, l2_reg=self.l2_reg))

        with tf.variable_scope("moe_layer"):
            moe_dnn = {}
            for name in self.loss_weight.keys():
                moe_dnn[name] = self._moe_layer(sharing_dnn, shared_expert_dnn + expert_dnn[name], scope=name)

        with tf.variable_scope("bayes_layer"):
            logits = bayes_layer(moe_dnn, is_training, self.specific_size,
                                 dropout=self.dropout, l2_reg=self.l2_reg)
            if match_logits is not None:
                for name in logits.keys():
                    logits[name] = [logits[name], match_logits]
            else:
                logits = {name: [value] for name, value in logits.items()}

        if self.use_transformer and self.use_transformer_logits:
            # 增加transformer_logits
            if self.logits_with_label:
                self._add_label_logits(inputs=transformer_logits, is_training=is_training, logits=logits,
                                       scope='transformer_logits')
            else:
                self._add_logits(inputs=transformer_logits, is_training=is_training, logits=logits,
                                 scope='transformer_logits')

        if self.use_linear_logits:
            # 增加线性层的logits，增加模型的泛化能力
            if self.logits_with_label:
                self._add_label_logits(inputs=dnn_input, is_training=is_training, logits=logits,
                                       scope="linear_logits")
            else:
                self._add_logits(inputs=dnn_input, is_training=is_training, logits=logits,
                                 scope="linear_logits")

        for name in logits.keys():
            # logits[name] = tf.concat(reduce(lambda x, y: x * y, logits[name]) + reduce(lambda x, y: x + y, logits[name]),
            #                          axis=-1)
            logits[name] = sum(logits[name])

        with tf.variable_scope("loss"):
            loss_dict = {}
            pred_dict = {}
            for name in self.loss_weight.keys():
                loss_dict[name], pred_dict[name] = compute_loss(logits[name], inputs[name], scope=name,
                                                                negative_decay=self.negative_decay, l2_reg=0.)

        # weight_sum = sum(self.loss_weight.values())
        # loss_weight = {k: v / weight_sum for k, v in self.loss_weight.items()}
        # total_loss = sum([loss_dict[key] * loss_weight[key] for key in loss_dict.keys()])
        total_loss = sum([loss_dict[key] for key in loss_dict.keys()])
        # total_loss = multi_loss_tradeoff(loss_dict)
        loss_dict['total_loss'] = total_loss

        label_dict = {name: inputs[name] for name in self.loss_weight.keys()}

        return loss_dict, pred_dict, label_dict

    def _moe_layer(self, sharing_dnn, specific_dnn, scope):
        with tf.variable_scope(scope):
            # 专家网络的维度
            specific_num = len(specific_dnn)
            # 门控：控制每个expert的权重
            gate_weight = tf.layers.dense(sharing_dnn, specific_num,
                                          kernel_initializer=slim.variance_scaling_initializer(),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            gate_weight = tf.nn.softmax(gate_weight)

            # [ [batch_size, dim], ....] -> [ [batch_size, 1, dim], ....] -> [batch_size, num, dim]
            specific_dnn = tf.concat([tf.expand_dims(dnn, axis=1) for dnn in specific_dnn], axis=1)
            # [batch_size, num, 1]
            gate_weight = tf.expand_dims(gate_weight, axis=-1)
            # [batch_size, dim]
            return tf.reduce_sum(specific_dnn * gate_weight, axis=1)

    def _features_fusion(self, emb_inputs, inputs):
        feed_inputs = tf.concat([tf.expand_dims(emb_inputs[name], axis=-2) for name in self.feed_features_list],
                                axis=-2)

        if len(feed_inputs.shape.as_list()) == 3:  # [batch_size, feat_num, dim]
            feed_weight = tf.nn.embedding_lookup(self.feed_weight_table, inputs['feedid'])
        elif len(feed_inputs.shape.as_list()) == 4:  # [batch_size, seq_len, feat_num, dim]
            feed_weight = tf.nn.embedding_lookup(self.feed_weight_table, inputs['feedid_history'])
        else:
            raise ValueError(feed_inputs)
        feed_weight = tf.nn.softmax(feed_weight)
        feed_weight = tf.expand_dims(feed_weight, axis=-1)

        output = tf.reduce_sum(feed_inputs * feed_weight, axis=-2)

        return output

    def _transformer_layer(self, his_feed, ctr_feed, history_seq_len, max_seq_len,
                           his_interval=None, interval_table=None):

        # token_type：代表是ctr还是history
        token_type_table = tf.get_variable("token_type_table",
                                           shape=[2, his_feed.shape.as_list()[-1]],
                                           initializer=tf.truncated_normal_initializer(stddev=0.02))
        # ctr放置在history前面，即第一个token_type_ids为1，其他为0
        token_type_ids = tf.sequence_mask(tf.minimum(1, history_seq_len), max_seq_len,
                                          dtype=tf.int32)

        transformer_inputs = tf.concat([tf.expand_dims(ctr_feed, axis=1), his_feed], axis=1)
        transformer_inputs = transformer_inputs + tf.nn.embedding_lookup(token_type_table, token_type_ids)

        if his_interval is not None and interval_table is not None:
            # his_interval最小值为0，为了让ctr_interval为0，所以his_interval+1
            # [batch_size,]
            ctr_interval = tf.minimum(0, history_seq_len)
            interval_ids = tf.concat([tf.expand_dims(ctr_interval, axis=1), his_interval + 1], axis=1)
            transformer_inputs = transformer_inputs + tf.nn.embedding_lookup(interval_table, interval_ids)

        # [batch_size, max_seq_len, dim]
        attention_mask = tf.sequence_mask(history_seq_len, max_seq_len,
                                          dtype=tf.int32)
        transformer_output = transformer_model(input_tensor=transformer_inputs,
                                               attention_mask=attention_mask,
                                               hidden_size=768,
                                               num_hidden_layers=6,
                                               num_hidden_groups=1,
                                               num_attention_heads=6,
                                               intermediate_size=1536,
                                               inner_group_num=1,
                                               intermediate_act_fn=gelu,
                                               hidden_dropout_prob=0.1,
                                               attention_probs_dropout_prob=0.1,
                                               initializer_range=0.02,
                                               do_return_all_layers=False)
        return transformer_output[:, 0, :], transformer_output[:, 1:, :]

    def _history_add_behaviors(self, his_feed, his_behaviors, mask):
        """

        :param his_feed: [batch_size, seq_len, dim]
        :param his_behaviors: [batch_size, seq_len, label_num]
        :param mask: [batch_size, seq_len]
        :return:
        """
        # [batch_size, seq_len, 1]
        his_behaviors_weight = tf.layers.dense(his_behaviors, 1,
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        # [batch_size, seq_len, dim]
        output = his_feed * his_behaviors_weight * tf.expand_dims(mask, axis=-1)

        return tf.reduce_sum(output, axis=-1)

    def _add_logits(self, inputs, is_training, logits, scope):
        linear_logits = tf.layers.dense(inputs, self.specific_size[-1],
                                        kernel_initializer=slim.variance_scaling_initializer(),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                        name=scope)
        if is_training:
            linear_logits = tf.nn.dropout(linear_logits, 1 - self.dropout)
        for name in logits.keys():
            logits[name].append(linear_logits)

    def _add_label_logits(self, inputs, is_training, logits, scope):
        for name in logits.keys():
            linear_logits = tf.layers.dense(inputs, self.specific_size[-1],
                                            kernel_initializer=slim.variance_scaling_initializer(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
                                            name=name+"_"+scope)
            if is_training:
                linear_logits = tf.nn.dropout(linear_logits, 1 - self.dropout)
            logits[name].append(linear_logits)
