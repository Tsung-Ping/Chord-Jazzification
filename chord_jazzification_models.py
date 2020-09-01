import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
from tensorflow.python.ops import array_ops
import numpy as np
import math
import time
import random
import pickle
import pretty_midi as pm
import os

triad_id_to_chroma = [[0 if i not in [0,4,7] else 1 for i in range(12)], [0 if i not in [0,3,7] else 1 for i in range(12)],
                      [0 if i not in [0,4,8] else 1 for i in range(12)], [0 if i not in [0,3,6] else 1 for i in range(12)],
                      [0 for _ in range(12)]] # for 'M', 'm', 'a', 'd', 'None'
chroma_table = np.array([np.roll(triad_id_to_chroma[i//12], i%12) if i < 48 else triad_id_to_chroma[4] for i in range(61)])
root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'X']
triad_type_list = ['M', 'm', 'a', 'd', 'X']

def _get_valid_length(duraion_data):
    valid_lens = []
    seq_len = duraion_data.shape[1]
    for sequence in duraion_data:
        try:
            valid_lens.append(np.where(sequence == 0)[0][0])
        except:
            valid_lens.append(seq_len)
    return np.array(valid_lens, dtype=np.int32)

def BLSTM_coloring(x_r, x_tr, x_d, y_len, dropout, hp):
    x_rtr = x_r + x_tr * 12 # triad id, major = {0-11}, minor = {12-23}, augmented = {24-35}, diminished = {36-47}
    onehot_embedding = tf.constant(chroma_table, dtype=tf.float32)
    x_onehot = tf.nn.embedding_lookup(onehot_embedding, x_rtr) # convert triad_type to one-hot vector
    input = x_onehot * tf.nn.tanh(x_d[:,:,tf.newaxis]) # mutiply one-hot vector by duration

    embedding_size = hp.n_units
    hidden_size = embedding_size
    with tf.variable_scope('Coloring'):
        with tf.name_scope('Input_embdedding'):
            if hp.with_em:
                input_embedded = tf.layers.dense(input, embedding_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
            else:
                input_embedded = input

        with tf.name_scope('LSTM_cells'):
            cell_fw = LSTMCell(num_units=hidden_size, name='cell_fw')
            cell_bw = LSTMCell(num_units=hidden_size, name='_cell_bw')
            cell_fw = DropoutWrapper(cell_fw, input_keep_prob=1 - dropout, output_keep_prob=1 - dropout)
            cell_bw = DropoutWrapper(cell_bw, input_keep_prob=1 - dropout, output_keep_prob=1 - dropout)

        with tf.name_scope('RNN'):
            (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                             cell_bw=cell_bw,
                                                                             inputs=input_embedded,
                                                                             sequence_length=y_len,
                                                                             dtype=tf.float32,
                                                                             time_major=False)
            hidden_states = tf.concat((output_fw, output_bw), axis=-1)

        with tf.name_scope('Output_projection'):
            inner = tf.layers.dense(hidden_states, hp.inner_b_classes + hp.inner_p_classes)
            inner_splits = tf.constant([hp.inner_b_classes, hp.inner_p_classes])
            b_logits, p_logits = array_ops.split(value=inner, num_or_size_splits=inner_splits, axis=2, name='split') # b_logits: 13 for classification, p_logits: 12 for binary regression

    return b_logits, p_logits

def BLSTM_voicing(x_b, x_p, x_d, y_len, dropout, hp):

    embedding_size = hp.n_units
    hidden_size = hp.n_units
    with tf.variable_scope('Voicing'):
        with tf.name_scope('Input_embdedding'):
            x_b_onehot = tf.one_hot(x_b, depth=hp.inner_b_classes)[:, :, :12]
            x_b_expand = tf.concat([tf.tile(x_b_onehot, [1, 1, 7]), tf.tile(tf.zeros_like(x_b_onehot), [1, 1, 2])], axis=2)[:, :, 9:97] # 88-D piano-roll vector
            x_p_expand = tf.concat([tf.tile(tf.zeros_like(x_p), [1, 1, 2]), tf.tile(x_p, [1, 1, 7])], axis=2)[:, :, 9:97] # 88-D piano-roll vector
            pianoroll_mask = tf.logical_or(tf.cast(x_b_expand, tf.bool), tf.cast(x_p_expand, tf.bool)) # [batch, n_steps, 88]
            pianoroll_mask_float = tf.cast(pianoroll_mask, tf.float32) # [batch, n_steps, 88]
            input = tf.concat([tf.cast(x_p, tf.float32), x_b_onehot], axis=2) * tf.nn.tanh(x_d[:,:,tf.newaxis])
            if hp.with_em:
                input_embedded = tf.layers.dense(input, embedding_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
            else:
                input_embedded = input

        with tf.name_scope('LSTM_cells'):
            cell_fw = LSTMCell(num_units=hidden_size, name='cell_fw')
            cell_bw = LSTMCell(num_units=hidden_size, name='_cell_bw')
            cell_fw = DropoutWrapper(cell_fw, input_keep_prob=1 - dropout, output_keep_prob=1 - dropout)
            cell_bw = DropoutWrapper(cell_bw, input_keep_prob=1 - dropout, output_keep_prob=1 - dropout)

        with tf.name_scope('RNN'):
            (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                             cell_bw=cell_bw,
                                                                             inputs=input_embedded,
                                                                             sequence_length=y_len,
                                                                             dtype=tf.float32,
                                                                             time_major=False)
            hidden_states = tf.concat((output_fw, output_bw), axis=-1) # shape = [batch, n_steps, 2*n_units]

        with tf.name_scope('Output_projection'):
            v_logits = tf.layers.dense(hidden_states, hp.output_v_classes) # voicing, [batch, n_steps, output_v_classes]
            n_splits = tf.constant([3, 12, 12, 12, 12, 12, 12, 12, 1]) #  split by octaves
            v_logits_split = array_ops.split(value=v_logits, num_or_size_splits=n_splits, axis=2, name='v_logits_split')
            v_logits_split[0] = tf.concat([tf.zeros([tf.shape(v_logits)[0], tf.shape(v_logits)[1], 9]), v_logits_split[0]], axis=2) # pad the first octave
            v_logits_split[-1] = tf.concat([v_logits_split[-1], tf.zeros([tf.shape(v_logits)[0], tf.shape(v_logits)[1], 11])], axis=2) # pad the last octave
            b_average_factors = tf.constant([5,5,5,5,5,5,5,5,5,6,6,6], dtype=tf.float32) # average by number of octaves
            b_logits = tf.add_n(v_logits_split[:6]) / b_average_factors[tf.newaxis, tf.newaxis, :] # bass
            p_average_factors = tf.constant([6,5,5,5,5,5,5,5,5,5,5,5], dtype=tf.float32) # average by number of octaves
            p_logits = tf.add_n(v_logits_split[-6:]) / p_average_factors[tf.newaxis, tf.newaxis, :] # pitch classes

    return v_logits, b_logits, p_logits, pianoroll_mask_float

def BLSTM_end2end(x_r, x_tr, x_d, y_len, dropout, hp):
    '''One-stage chord jazzification using BLSTM'''

    x_rtr = x_r + x_tr * 12 # triad id, major = {0-11}, minor = {12-23}, augmented = {24-35}, diminished = {36-47}
    onehot_embedding = tf.constant(chroma_table, dtype=tf.float32)
    x_onehot = tf.nn.embedding_lookup(onehot_embedding, x_rtr) # convert triad_type to one-hot vector
    input = x_onehot * tf.nn.tanh(x_d[:, :, tf.newaxis]) # mutiply vector by duration

    embedding_size = hp.n_units*2
    hidden_size = embedding_size
    n_layers = 2
    with tf.name_scope('Input_embdedding'):
        if hp.with_em:
            input_embedded = tf.layers.dense(input, embedding_size, kernel_initializer=tf.contrib.layers.xavier_initializer()) # dense vectors
        else:
            input_embedded = input # one-hot vectors

    with tf.name_scope('Multilayer_cells'):
        def cell_drop(hidden_size):
            cell = LSTMCell(num_units=hidden_size)
            return DropoutWrapper(cell, input_keep_prob=1 - dropout, output_keep_prob=1.0)
        cells_fw = tf.nn.rnn_cell.MultiRNNCell([cell_drop(hidden_size) for _ in range(n_layers)])
        cells_bw = tf.nn.rnn_cell.MultiRNNCell([cell_drop(hidden_size) for _ in range(n_layers)])

    with tf.name_scope('Multilayer_RNN'):
        (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cells_fw,
                                                                         cell_bw=cells_bw,
                                                                         inputs=input_embedded,
                                                                         sequence_length=y_len,
                                                                         dtype=tf.float32,
                                                                         time_major=False)
        hidden_states = tf.concat((output_fw, output_bw), axis=-1)

    with tf.name_scope('Output_projection'):
        v_logits = tf.layers.dense(hidden_states, hp.output_v_classes)

    return v_logits

def _get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    """Return positional encoding. (see get_timing_signal_1d in https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py)
    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.
    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position
    Returns:
      Tensor with shape [length, hidden_size]
    """
    position = tf.cast(tf.range(length) + start_index, tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / tf.maximum(tf.cast(num_timescales, tf.float32) - 1, 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(hidden_size, 2)]])
    signal = tf.reshape(signal, [1, length, hidden_size])
    return signal

def _multihead_attention(queries, keys, num_units=None, n_heads=8, seq_mask=None, forward=False,
                        dropout_rate=0, is_training=True, scope="multihead_attention", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Project first, and then split
        # Linear projections
        Q = tf.layers.dense(queries, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()) # (N, T_k, C)

        # Split and concat (multihead)
        Q_ = tf.concat(tf.split(Q, n_heads, axis=2), axis=0) # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, n_heads, axis=2), axis=0) # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, n_heads, axis=2), axis=0) # (h*N, T_k, C/h)

        # Multiplication (Compute dot similarity)
        outputs = tf.matmul(Q_, K_, transpose_b=True) # QK^T, shape=(h*N, T_q, T_k)

        # Scale by hidden size
        outputs = outputs / (K_.get_shape().as_list()[-1]**0.5) # QK^T/sqrt(d_k), shape=(h*N, T_q, T_k)

        # Key Masking
        if seq_mask is not None:
            key_mask = tf.tile(tf.expand_dims(seq_mask, 1), [n_heads, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
            paddings = tf.ones_like(outputs) * (-2**32 +1) # set padded cells to a value close to -Infinity, so that their contributions are just negligible.
            outputs = tf.where(tf.equal(key_mask, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        """only for self attention"""
        if forward:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # lower-triangualr part of the attention matrix, (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks) *(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs, axis=2) # softmax(QK^T/sqrt(k_d)), shape=(h*N, T_q, T_k)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # softmax(QK^T/sqrt(k_d))V, ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, n_heads, axis=0), axis=2) # Concat(head 1 , ..., head h ), shape=(N, T_q, C)

        # Head projection
        outputs = tf.layers.dense(outputs, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()) # Concat(head 1 , ..., head h )W^O

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training) # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = _normalize(outputs) # (N, T_q, C)

    return outputs


def _feedforward(inputs, num_units=[2048, 512], activation_function=tf.nn.relu, dropout_rate=0,
                is_training=True, residual_connection=True, scope="feedforward", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": activation_function, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

        if residual_connection:
            # Residual connection
            outputs += inputs

        # Normalize
        outputs = _normalize(outputs)

    return outputs

def _normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    '''Applies layer normalization'''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta_bias", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def Attention_coloring(x_r, x_tr, x_d, y_len, dropout, is_training, hp):
    embedding_size = hp.n_units
    hidden_size = embedding_size
    with tf.variable_scope('Coloring'):
        with tf.name_scope('Input_embdedding'):
            x_rtr = x_r + x_tr * 12 # triad id, major = {0-11}, minor = {12-23}, augmented = {24-35}, diminished = {36-47}
            onehot_embedding = tf.constant(chroma_table, dtype=tf.float32)
            x_onehot = tf.nn.embedding_lookup(onehot_embedding, x_rtr) # convert triad_type to one-hot vector, shape = [batch, n_steps, 12]
            input = x_onehot * tf.nn.tanh(x_d[:, :, tf.newaxis]) # mutiply one-hot vector by duration
            input_embedded = tf.layers.dense(input, embedding_size, kernel_initializer=tf.contrib.layers.xavier_initializer())

            seq_mask = tf.sequence_mask(lengths=y_len, maxlen=tf.shape(input)[1], dtype=tf.float32)  # seqence mask by which paddings are excluded from attention, [batch, n_steps]

        with tf.name_scope('Positional_Encoding'):
            input_embedded += _get_position_encoding(tf.shape(input_embedded)[1], hidden_size)

        # Dropout
        input_embedded = tf.layers.dropout(input_embedded, rate=dropout, training=is_training)

        with tf.name_scope('Multihead_Attention'):
            for i in range(2):
                with tf.variable_scope("n_blocks_{}".format(i)):
                    # Multihead Attention (self-attention)
                    input_embedded = _multihead_attention(queries=input_embedded,
                                                         keys=input_embedded,
                                                         num_units=hidden_size,
                                                         n_heads=8,
                                                         seq_mask=seq_mask,
                                                         forward=False,
                                                         dropout_rate=dropout,
                                                         is_training=is_training,
                                                         scope="self_attention")

                    # Feed Forward
                    input_embedded = _feedforward(input_embedded, num_units=[hidden_size*4, hidden_size], activation_function=tf.nn.relu, dropout_rate=dropout, is_training=is_training)

    with tf.name_scope('Output_projection'):
        inner = tf.layers.dense(input_embedded, hp.inner_b_classes + hp.inner_p_classes)
        inner_splits = tf.constant([hp.inner_b_classes, hp.inner_p_classes])
        b_logits, p_logits = array_ops.split(value=inner, num_or_size_splits=inner_splits, axis=2, name='split') # y_b: 13 for classification, y_p: 12 for binary regression

    return b_logits, p_logits

def Attention_voicing(x_b, x_p, x_d, y_len, dropout, is_training, hp):
    embedding_size = hp.n_units
    hidden_size = hp.n_units
    with tf.variable_scope('Voicing'):
        with tf.name_scope('Input_embdedding'):
            x_b_onehot = tf.one_hot(x_b, depth=hp.inner_b_classes)[:, :, :12]
            x_b_expand = tf.concat([tf.tile(x_b_onehot, [1, 1, 7]), tf.tile(tf.zeros_like(x_b_onehot), [1, 1, 2])], axis=2)[:, :, 9:97] # [batch, n_steps, 88]
            x_p_expand = tf.concat([tf.tile(tf.zeros_like(x_p), [1, 1, 2]), tf.tile(x_p, [1, 1, 7])], axis=2)[:, :, 9:97] # [batch, n_steps, 88]
            pianoroll_mask = tf.logical_or(tf.cast(x_b_expand, tf.bool), tf.cast(x_p_expand, tf.bool)) # [batch, n_steps, 88]
            pianoroll_mask_float = tf.cast(pianoroll_mask, tf.float32) # [batch, n_steps, 88]
            input = tf.concat([tf.cast(x_p, tf.float32), x_b_onehot], axis=2) * tf.nn.tanh(x_d)[:, :, tf.newaxis]
            input_embedded = tf.layers.dense(input, embedding_size, kernel_initializer=tf.contrib.layers.xavier_initializer()) # [batch, n_steps, n_units]

            seq_mask = tf.sequence_mask(lengths=y_len, maxlen=tf.shape(input)[1], dtype=tf.float32) # seqence mask by which paddings are excluded from attention, [batch, n_steps]

        with tf.name_scope('Positional_Encoding'):
            input_embedded += _get_position_encoding(tf.shape(input_embedded)[1], hidden_size)

        # Dropout
        input_embedded = tf.layers.dropout(input_embedded, rate=dropout, training=is_training)

        with tf.name_scope('Multihead_Attention'):
            for i in range(2):
                with tf.variable_scope("n_blocks_{}".format(i)):
                    # Multihead Attention (self-attention)
                    input_embedded = _multihead_attention(queries=input_embedded,
                                                         keys=input_embedded,
                                                         num_units=hidden_size,
                                                         n_heads=8,
                                                         seq_mask=seq_mask,
                                                         dropout_rate=dropout,
                                                         is_training=is_training,
                                                         scope="self_attention")
                    # Feed Forward
                    input_embedded = _feedforward(input_embedded, num_units=[hidden_size*4, hidden_size], dropout_rate=dropout, is_training=is_training)

    with tf.name_scope('Output_projection'):
        v_logits = tf.layers.dense(input_embedded, hp.output_v_classes) # [batch, n_steps, output_v_classes]
        n_splits = tf.constant([3, 12, 12, 12, 12, 12, 12, 12, 1]) # split by octaves
        v_logits_split = array_ops.split(value=v_logits, num_or_size_splits=n_splits, axis=2, name='v_logits_split')
        n_batch, n_steps = tf.shape(v_logits)[0], tf.shape(v_logits)[1]
        v_logits_split[0] = tf.concat([tf.zeros([n_batch, n_steps, 9]), v_logits_split[0]], axis=2) # pad the first octave
        v_logits_split[-1] = tf.concat([v_logits_split[-1], tf.zeros([n_batch, n_steps, 11])], axis=2) # pad the last octave
        b_average_factors = tf.constant([5,5,5,5,5,5,5,5,5,6,6,6], dtype=tf.float32) # average by number of octaves
        b_logits = tf.add_n(v_logits_split[:6]) / b_average_factors[tf.newaxis, tf.newaxis, :] # bass
        p_average_factors = tf.constant([6,5,5,5,5,5,5,5,5,5,5,5], dtype=tf.float32) # average by number of octaves
        p_logits = tf.add_n(v_logits_split[-6:]) / p_average_factors[tf.newaxis, tf.newaxis, :] # pitch calsses

    return v_logits, b_logits, p_logits, pianoroll_mask_float

def _F1_score(predicted, actual):
    TP = tf.count_nonzero(predicted * actual)
    # TN = tf.count_nonzero((predicted - 1) * (actual - 1))
    FP = tf.count_nonzero(predicted * (actual - 1))
    FN = tf.count_nonzero((predicted - 1) * actual)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    F1 = tf.cond(tf.is_nan(F1), lambda: tf.constant(0.0, dtype=tf.float64), lambda: F1)
    return precision, recall, F1

def _compute_SSM(input):
    input_norm = tf.nn.l2_normalize(tf.cast(input, tf.float32), axis=-1)
    return tf.matmul(input_norm, input_norm, transpose_b=True)

def _SSM_masking(SSM, seq_mask=None):
    mask = tf.matmul(seq_mask[:, :, tf.newaxis], seq_mask[:, :, tf.newaxis], transpose_b=True)
    mask = tf.cast(mask, tf.bool)
    return tf.boolean_mask(SSM, mask)

def load_data(data_dir, validation_set_id):
    # 4-fold cross validation, with validation_set_id = {0, 1, 2, 3}
    if validation_set_id > 3 or validation_set_id < 0:
        print('Invalid validation_set_id.')
        exit(1)

    '''0-795: shift = -4
         796-1591: shift = -3
         ...
         3184-3979: shift = 0'''

    print("Load data...")
    with open(data_dir, 'rb') as file:
        data = pickle.load(file)

    data_reshape = np.stack(np.split(data, indices_or_sections=10), axis=2) # [n_sequences, n_steps, shift]
    data_reshape_split = np.split(data_reshape, indices_or_sections=4, axis=0) # split into 4 sets
    testing_set = data_reshape_split[validation_set_id][:,:,4] # 4 for the original data without transposition
    training_set =np.concatenate([x for i, x in enumerate(data_reshape_split) if i != validation_set_id], axis=0)
    training_set = np.squeeze(np.concatenate(np.split(training_set, indices_or_sections=10, axis=2), axis=0), axis=2)

    print('Data Info:')
    print('data.shape = (7960, 67) # [n_sequences, n_steps], with fields [\'root\', \'triad_type\', \'duration\', \'bass\', \'treble\', \'piano_vector\']')
    print('root: 0-11 for C-B; 12 for \'None\' and padding')
    print('triad_type: 0-3 for M, m, a, d; 4 for \'None\' and padding')
    print('duration: float; 0 for padding')
    print('bass: 0-11 for C-B; 12 for \'None\' and padding')
    print('treble: chroma vector; all zeros for \'None\' and padding')
    print('piano_vector: 88-d vector; all zeros for \'None\' and padding')

    data = {'train': training_set, 'test': testing_set}

    return data

def load_data_without_separation(data_dir):
    '''without cross validation'''
    print("Load data...")
    with open(data_dir, 'rb') as file:
        data = pickle.load(file)

    print('Data Info:')
    print('data.shape = (7960, 67) # [n_sequences, n_steps], with fields [\'root\', \'triad_type\', \'duration\', \'bass\', \'treble\', \'piano_vector\']')
    print('root: 0-11 for C-B; 12 for \'None\' and padding')
    print('triad_type: 0-3 for M, m, a, d; 4 for \'None\' and padding')
    print('duration: float; 0 for padding')
    print('bass: 0-11 for C-B; 12 for \'None\' and padding')
    print('treble: chroma vector; all zeros for \'None\' and padding')
    print('piano_vector: 88-d vector; all zeros for \'None\' and padding')
    return data

def train_coloring_model(data_dir, hp):
    # Load training data
    data = load_data_without_separation(data_dir)

    X = {'coloring': {'root': data['root'],
                    'triad_type': data['triad_type'],
                    'duration': data['duration']},
         'voicing': {'bass': data['bass'],
                   'treble': np.reshape([chroma for chroma in data['treble'].flatten()], newshape=[data.shape[0], data.shape[1], 12]),
                    'duration': data['duration']}}
    Y = {'coloring': {'bass': data['bass'],
                    'treble': np.reshape([chroma for chroma in data['treble'].flatten()], newshape=[data.shape[0], data.shape[1], 12]),
                    'lens': _get_valid_length(data['duration'])},
         'voicing': {'piano_vector': np.reshape([vector for vector in data['piano_vector'].flatten()], newshape=[data.shape[0], data.shape[1], 88]),
                   'lens': _get_valid_length(data['duration'])}}

    n_sequences = data.shape[0]
    n_iterations_per_epoch = math.ceil(n_sequences / hp.batch_size)
    print('number of training sequences =', n_sequences)
    print('n_iterations_per_epoch=', n_iterations_per_epoch)
    print(hp) # hyperparameters
    print('sequential_model =', hp.sequential_model)

    with tf.Graph().as_default() as g_coloring:
        with tf.name_scope('Chord_jazzification_coloring'):
            # Placeholders
            x_r = tf.placeholder(tf.int32, [None, hp.n_steps], name="root")
            x_tr = tf.placeholder(tf.int32, [None, hp.n_steps], name="triad_type")
            x_d = tf.placeholder(tf.float32, [None, hp.n_steps], name="duration")
            y_b = tf.placeholder(tf.int32, [None, hp.n_steps], name="bass")
            y_p = tf.placeholder(tf.int32, [None, hp.n_steps, 12], name="pitch_classes")
            y_len = tf.placeholder(tf.int32, [None], name="seq_lens")
            dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
            global_step = tf.placeholder(dtype=tf.int32, name='global_step')
            is_training = tf.placeholder(dtype=tf.bool, name='is_training')

            if hp.sequential_model == 'blstm':
                b_logits, p_logits = BLSTM_coloring(x_r, x_tr, x_d, y_len, dropout, hp)
            elif hp.sequential_model == 'mhsa':
                b_logits, p_logits = Attention_coloring(x_r, x_tr, x_d, y_len, dropout, is_training, hp)
            else:
                print('Invalid model name.')
                exit(1)

        with tf.name_scope('Loss'):
            seq_mask = tf.sequence_mask(lengths=y_len, maxlen=tf.shape(x_r)[1], dtype=tf.float32) # seqence mask by which paddings are filtered, [batch, n_steps]

            # Cross entropy
            ce_b = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_b, hp.inner_b_classes), logits=b_logits, weights=seq_mask, label_smoothing=0.0) # bass
            ce_p = hp.beta_ce_p * tf.losses.sigmoid_cross_entropy(multi_class_labels=y_p, logits=p_logits, weights=seq_mask[:, :, tf.newaxis], label_smoothing=0.0) # pitch calsses

            # L2 norm regularization
            vars = tf.trainable_variables()
            L2_regularizer = hp.coloring_beta_L2 * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])

            #  Total loss
            loss = ce_b + ce_p + L2_regularizer
        tf.summary.scalar('Loss_total', loss)
        tf.summary.scalar('Loss_bass', ce_b)
        tf.summary.scalar('Loss_pitch_classes', ce_p)
        tf.summary.scalar('Loss_L2', L2_regularizer)

        with tf.name_scope('Optimization'):
            # apply learning rate decay
            learning_rate = tf.train.exponential_decay(learning_rate=hp.initial_learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=n_iterations_per_epoch,
                                                       decay_rate=0.96,
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               beta1=0.9,
                                               beta2=0.98,
                                               epsilon=1e-9)

            # Apply gradient clipping
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (grad, var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs)

        with tf.name_scope('Evaluation'):
            # Bass
            pred_b = tf.argmax(b_logits, axis=2, output_type=tf.int32)
            mask_b = tf.cast(tf.tile(tf.expand_dims(seq_mask, 2), [1, 1, hp.inner_b_classes]), tf.bool)
            pred_b_mask = tf.boolean_mask(tf.one_hot(pred_b, depth=hp.inner_b_classes), mask_b)
            y_b_mask = tf.boolean_mask(tf.one_hot(y_b, depth=hp.inner_b_classes), mask_b)
            _, _, F1_b = _F1_score(pred_b_mask, y_b_mask)

            # Pitch classes
            pred_p = tf.cast(tf.round(tf.sigmoid(p_logits)), tf.int32)
            mask_p = tf.cast(tf.tile(tf.expand_dims(seq_mask, 2), [1, 1, hp.inner_p_classes]), tf.bool)
            pred_p_mask = tf.boolean_mask(pred_p, mask_p)
            y_p_mask = tf.boolean_mask(y_p, mask_p)
            _, _, F1_p = _F1_score(pred_p_mask, y_p_mask)
        tf.summary.scalar('F1_bass', F1_b)
        tf.summary.scalar('F1_pitch_classes', F1_p)

        graph_location = 'model\\'
        train_writer = tf.summary.FileWriter(graph_location + '\\train')
        print('Saving graph to: %s' % graph_location)
        merged = tf.summary.merge_all()
        train_writer.add_graph(tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=1)

        # Train for a fixed number of iterations
        print('train the coloring model...')
        with tf.Session(graph=g_coloring) as sess:
            sess.run(tf.global_variables_initializer())
            startTime = time.time() # start time of training
            best_scores = [0.0, 0.0]
            in_succession = 0
            best_epoch = 0
            for step in range(hp.training_steps):
                # Training
                if step == 0:
                    indices = range(n_sequences)
                    batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]

                # if step % n_iterations_per_epoch == 0 and step > 0:
                if step / n_iterations_per_epoch > 5:
                    # Shuffle training data
                    indices = random.sample(range(n_sequences), n_sequences)
                    batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]

                batch = (X['coloring']['root'][batch_indices[step % len(batch_indices)]],
                         X['coloring']['triad_type'][batch_indices[step % len(batch_indices)]],
                         X['coloring']['duration'][batch_indices[step % len(batch_indices)]],
                         Y['coloring']['bass'][batch_indices[step % len(batch_indices)]],
                         Y['coloring']['treble'][batch_indices[step % len(batch_indices)]],
                         Y['coloring']['lens'][batch_indices[step % len(batch_indices)]])

                train_run_list = [train_op, merged, pred_b, pred_p, loss, ce_b, ce_p, L2_regularizer, F1_b, F1_p, seq_mask]
                train_feed_fict = {x_r: batch[0],
                                   x_tr: batch[1],
                                   x_d: batch[2],
                                   y_b: batch[3],
                                   y_p: batch[4],
                                   y_len: batch[5],
                                   dropout: hp.drop,
                                   is_training: True,
                                   global_step: step + 1}
                _, train_summary, train_b_pred, train_p_pred, train_loss, train_ce_b, train_ce_p, train_L2, train_F1_b, train_F1_p, mask = sess.run(train_run_list, feed_dict=train_feed_fict)

                # Display training log
                if step % n_iterations_per_epoch == 0:
                    if step == 0:
                        print('*~ ce_b %.4f, ce_p %.4f, L2 %.4f ~*' % (train_ce_b, train_ce_p, train_L2))
                    train_writer.add_summary(train_summary, step)
                    print(
                        "------ step %d, epoch %d: train_loss %.4f (b %.4f, p %.4f, L2 %.4f), train_F1: (b %.4f, p %.4f) ------"
                        % (step, step // n_iterations_per_epoch, train_loss, train_ce_b, train_ce_p, train_L2, train_F1_b, train_F1_p))
                    print('len'.ljust(4, ' '), batch[5][0])
                    print('mask'.ljust(4, ' '), ''.join(['T'.rjust(4, ' ') if m else 'F'.rjust(4, ' ') for m in mask[0]]))
                    print('x_r'.ljust(4, ' '), ''.join([root_list[b].rjust(4, ' ') for b in batch[0][0]]))
                    print('x_tr'.ljust(4, ' '), ''.join([triad_type_list[b].rjust(4, ' ') for b in batch[1][0]]))
                    print('x_d'.ljust(4, ' '), ''.join([str(round(b, 1))[:3].rjust(4, ' ') for b in batch[2][0]]))
                    print('y_b'.ljust(4, ' '), ''.join([root_list[b].rjust(4, ' ') for b in batch[3][0]]))
                    print('p_b'.ljust(4, ' '), ''.join([root_list[b].rjust(4, ' ') for b in train_b_pred[0]]))
                    print('y_p'.ljust(4, ' '), ''.join([str(b) for b in batch[4][0, :8]]))
                    print('p_p'.ljust(4, ' '), ''.join([str(b) for b in train_p_pred[0, :8]]))

                    # Check if early stop
                    if train_F1_b + train_F1_p > sum(best_scores):
                        best_scores = [train_F1_b, train_F1_p]
                        best_epoch = step // n_iterations_per_epoch
                        in_succession = 0
                        # Save parameters
                        print('*save parameters...')
                        saver.save(sess, graph_location + '\\coloring_model.ckpt')
                    else:
                        in_succession += 1
                        if in_succession > hp.n_in_succession:
                            print('Early stop.')
                            break

            elapsed_time = time.time() - startTime
            np.set_printoptions(precision=4)
            print('Training result of coloring:')
            print('training time = %.2f hr' % (elapsed_time / 3600))
            print('best training score =', np.round(best_scores, 4))
            print('best epoch = ', best_epoch)


def train_voicing_model(data_dir, hp):
    # Load training data
    data = load_data_without_separation(data_dir)

    X = {'coloring': {'root': data['root'],
                    'triad_type': data['triad_type'],
                    'duration': data['duration']},
         'voicing': {'bass': data['bass'],
                   'treble': np.reshape([chroma for chroma in data['treble'].flatten()], newshape=[data.shape[0], data.shape[1], 12]),
                   'duration': data['duration']}}
    Y = {'coloring': {'bass': data['bass'],
                    'treble': np.reshape([chroma for chroma in data['treble'].flatten()], newshape=[data.shape[0], data.shape[1], 12]),
                    'lens': _get_valid_length(data['duration'])},
         'voicing': {'piano_vector': np.reshape([vector for vector in data['piano_vector'].flatten()], newshape=[data.shape[0], data.shape[1], 88]),
                   'lens': _get_valid_length(data['duration'])}}

    n_sequences = data.shape[0]
    n_iterations_per_epoch = math.ceil(n_sequences / hp.batch_size)
    print('number of training sequences =', n_sequences)
    print('n_iterations_per_epoch=', n_iterations_per_epoch)
    print(hp)  # hyperparameters
    print('sequential_model =', hp.sequential_model)

    with tf.Graph().as_default() as g_voicing:
        with tf.name_scope('Voicing_model'):
            # Placeholders
            x_b = tf.placeholder(tf.int32, [None, hp.n_steps], name="bass")
            x_p = tf.placeholder(tf.int32, [None, hp.n_steps, 12], name="pitch_calsses")
            x_d = tf.placeholder(tf.float32, [None, hp.n_steps], name="duration")
            y_v = tf.placeholder(tf.int32, [None, hp.n_steps, 88], name="voicing")
            y_len = tf.placeholder(tf.int32, [None], name="seq_lens")
            dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
            is_training = tf.placeholder(dtype=tf.bool, name="is_training")
            global_step = tf.placeholder(dtype=tf.int32, name='global_step')

            if hp.sequential_model == 'blstm':
                v_logits, b_logits, p_logits, pianoroll_mask_float = BLSTM_voicing(x_b, x_p, x_d, y_len, dropout, hp)
            elif hp.sequential_model == 'mhsa':
                v_logits, b_logits, p_logits, pianoroll_mask_float = Attention_voicing(x_b, x_p, x_d, y_len, dropout, is_training, hp)
            else:
                print('Invalid model name.')
                exit(1)

        with tf.name_scope('Loss'):
            seq_mask = tf.sequence_mask(lengths=y_len, maxlen=tf.shape(x_b)[1], dtype=tf.float32)  # seqence mask by which paddings are filtered, [batch, n_steps]

            # Voicing
            if hp.with_mask:
                ce_v = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_v, logits=v_logits, weights=pianoroll_mask_float, label_smoothing=0.0)
            else:
                ce_v = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_v, logits=v_logits, weights=seq_mask[:, :, tf.newaxis], label_smoothing=0.0)

            # Bass and pitch classes (in terms of chroma vector)
            ce_chroma = 0.5 * tf.losses.mean_squared_error(labels=tf.one_hot(x_b, depth=hp.inner_b_classes)[:, :, :12], predictions=tf.sigmoid(b_logits), weights=seq_mask[:, :, tf.newaxis]) \
                        + 0.5 * tf.losses.sigmoid_cross_entropy(multi_class_labels=x_p, logits=p_logits, weights=seq_mask[:, :, tf.newaxis])

            # L2 norm regularization
            vars = tf.trainable_variables()
            L2_regularizer = hp.voicing_beta_L2 * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])

            # Total loss
            loss = ce_v + ce_chroma + L2_regularizer
        tf.summary.scalar('Loss_total', loss)
        tf.summary.scalar('Loss_voicing', ce_v)
        tf.summary.scalar('Loss_chroma', ce_chroma)
        tf.summary.scalar('Loss_L2', L2_regularizer)

        with tf.name_scope('Optimization'):
            # apply learning rate decay
            learning_rate = tf.train.exponential_decay(learning_rate=hp.initial_learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=n_iterations_per_epoch,
                                                       decay_rate=0.96,
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               beta1=0.9,
                                               beta2=0.98,
                                               epsilon=1e-9)

            # Apply gradient clipping
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (grad, var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs)

        with tf.name_scope('Evaluation'):
            # Voicing
            pred_v = tf.round(tf.sigmoid(v_logits)) * pianoroll_mask_float
            pred_v = tf.cast(pred_v, tf.int32)
            mask_bool = tf.cast(pianoroll_mask_float, tf.bool)
            pred_v_mask = tf.boolean_mask(pred_v, mask_bool)
            y_v_mask = tf.boolean_mask(y_v, mask_bool)
            P_v, R_v, F1_v = _F1_score(pred_v_mask, y_v_mask)

            # Self similarity matrix
            pred_v_SSM = _compute_SSM(pred_v)
            y_v_SSM = _compute_SSM(y_v)
            pred_v_SSM_mask = _SSM_masking(pred_v_SSM, seq_mask=seq_mask)
            y_v_SSM_mask = _SSM_masking(y_v_SSM, seq_mask=seq_mask)
            SSM_score = 1.0 - tf.reduce_mean(tf.abs(pred_v_SSM_mask - y_v_SSM_mask))
        tf.summary.scalar('F1_voicing', F1_v)
        tf.summary.scalar('Precision_voicing', P_v)
        tf.summary.scalar('Recall_voicing', R_v)
        tf.summary.scalar('SSM_score', SSM_score)

        graph_location = 'model\\'
        train_writer = tf.summary.FileWriter(graph_location + '\\train')
        print('Saving graph to: %s' % graph_location)
        merged = tf.summary.merge_all()
        train_writer.add_graph(tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=1)

        # Train for a fixed number of iterations
        print('train the voicing model...')
        with tf.Session(graph=g_voicing) as sess:
            sess.run(tf.global_variables_initializer())
            startTime = time.time()  # start time of training
            best_score = 0.0
            in_succession = 0
            best_epoch = 0
            for step in range(hp.training_steps):
                # Training
                if step == 0:
                    indices = range(n_sequences)
                    batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]

                # if step % n_iterations_per_epoch == 0 and step > 0:
                if step / n_iterations_per_epoch > 5:
                    # Shuffle training data
                    indices = random.sample(range(n_sequences), n_sequences)
                    batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]

                batch = (X['voicing']['bass'][batch_indices[step % len(batch_indices)]],
                         X['voicing']['treble'][batch_indices[step % len(batch_indices)]],
                         X['voicing']['duration'][batch_indices[step % len(batch_indices)]],
                         Y['voicing']['piano_vector'][batch_indices[step % len(batch_indices)]],
                         Y['voicing']['lens'][batch_indices[step % len(batch_indices)]])

                train_run_list = [train_op, merged, pred_v, pianoroll_mask_float, loss, ce_v, ce_chroma, L2_regularizer,
                                  P_v, R_v, F1_v, SSM_score]
                train_feed_fict = {x_b: batch[0],
                                   x_p: batch[1],
                                   x_d: batch[2],
                                   y_v: batch[3],
                                   y_len: batch[4],
                                   dropout: hp.drop,
                                   is_training: True,
                                   global_step: step + 1}
                _, train_summary, train_v_pred, train_mask, train_loss, train_ce_v, train_ce_chroma, train_L2, train_P_v, train_R_v, train_F1_v, train_SSM_score = sess.run(train_run_list, feed_dict=train_feed_fict)

                # Display training log
                if step % n_iterations_per_epoch == 0:
                    if step == 0:
                        print('*~ ce_v %.4f, ce_chroma %.4f, L2 %.4f ~*' % (train_ce_v, train_ce_chroma, train_L2))
                    train_writer.add_summary(train_summary, step)
                    print(
                        "------ step %d, epoch %d: train_loss %.4f (v %.4f, chroma %.4f, L2 %.4f), train_scores (P %.4f, R %.4f, F1 %.4f, SSM %.4f) ------"
                        % (step, step // n_iterations_per_epoch, train_loss, train_ce_v, train_ce_chroma, train_L2, train_P_v, train_R_v, train_F1_v, train_SSM_score))
                    print('y_v'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v]) + ']' for v in batch[3][0, :2]]))
                    print('p_v'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v]) + ']' for v in train_v_pred[0, :2]]))
                    print('mask'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v.astype(np.int32)]) + ']' for v in train_mask[0, :2]]))

                    # Check if early stop
                    if train_F1_v > best_score:
                        best_score = train_F1_v
                        best_epoch = step // n_iterations_per_epoch
                        in_succession = 0
                        # Save parameters
                        print('*save parameters...')
                        saver.save(sess, graph_location + '\\voicing_model.ckpt')
                    else:
                        in_succession += 1
                        if in_succession > hp.n_in_succession:
                            break

            elapsed_time = time.time() - startTime
            print("Training info of voicing:")
            print('training time = %.2f hr' % (elapsed_time / 3600))
            print('best epoch = ', best_epoch)
            print('best valid score =', np.round(best_score, 4))

def train_chord_jazzification(data_dir, hp):
    # Train the two-stage chord jazzification model successively
    train_coloring_model(data_dir, hp)
    train_voicing_model(data_dir, hp)



def cross_validate_coloring_model(data_dir, hp):

    # Load training data
    print('Cross-validation: validation_set_id =', hp.validation_set_id)
    print('sequential_model =', hp.sequential_model)
    data = load_data(data_dir, hp.validation_set_id)
    X = {'train': data['train'][['root', 'triad_type', 'duration']],
         'test': data['test'][['root', 'triad_type', 'duration']]}
    Y = {'train': {'bass': data['train']['bass'],
                  'treble': np.reshape([chroma for chroma in data['train']['treble'].flatten()], newshape=[data['train'].shape[0], data['train'].shape[1], 12]),
                  'lens': _get_valid_length(data['train']['duration'])},
         'test': {'bass': data['test']['bass'],
                 'treble': np.reshape([chroma for chroma in data['test']['treble'].flatten()], newshape=[data['test'].shape[0], data['test'].shape[1], 12]),
                 'lens': _get_valid_length(data['test']['duration'])}}

    n_sequences = X['train']['root'].shape[0]
    n_iterations_per_epoch = math.ceil(n_sequences / hp.batch_size)
    print('number of training sequences =', n_sequences)
    print('number of testing sequences =', X['test']['root'].shape[0])
    print('n_iterations_per_epoch=', n_iterations_per_epoch)
    print(hp) # hyperparameters

    with tf.name_scope('Coloring_model'):
        # Placeholders
        x_r = tf.placeholder(tf.int32, [None, hp.n_steps], name="root")
        x_tr = tf.placeholder(tf.int32, [None, hp.n_steps], name="triad_type")
        x_d = tf.placeholder(tf.float32, [None, hp.n_steps], name="duration")
        y_b = tf.placeholder(tf.int32, [None, hp.n_steps], name="bass")
        y_p = tf.placeholder(tf.int32, [None, hp.n_steps, 12], name="pitch_classes")
        y_len = tf.placeholder(tf.int32, [None], name="seq_lens")
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        if hp.sequential_model == 'blstm':
            b_logits, p_logits = BLSTM_coloring(x_r, x_tr, x_d, y_len, dropout, hp)
        elif hp.sequential_model == 'mhsa':
            b_logits, p_logits = Attention_coloring(x_r, x_tr, x_d, y_len, dropout, is_training, hp)
        else:
            print('Invalid model name.')
            exit(1)

    with tf.name_scope('Loss'):
        seq_mask = tf.sequence_mask(lengths=y_len, maxlen=tf.shape(x_r)[1], dtype=tf.float32) # seqence mask by which paddings are filtered, [batch, n_steps]

        # Cross entropy
        ce_b = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_b, hp.inner_b_classes), logits=b_logits, weights=seq_mask, label_smoothing=0.0) # bass
        ce_p = hp.beta_ce_p * tf.losses.sigmoid_cross_entropy(multi_class_labels=y_p, logits=p_logits, weights=seq_mask[:, :, tf.newaxis], label_smoothing=0.0) # pitch calsses

        # L2 norm regularization
        vars = tf.trainable_variables()
        L2_regularizer = hp.coloring_beta_L2 * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])

        # Total loss
        loss = ce_b + ce_p + L2_regularizer

    tf.summary.scalar('Loss_total', loss)
    tf.summary.scalar('Loss_bass', ce_b)
    tf.summary.scalar('Loss_pitch_classes', ce_p)
    tf.summary.scalar('Loss_L2', L2_regularizer)

    with tf.name_scope('Optimization'):
        # apply learning rate decay
        learning_rate = tf.train.exponential_decay(learning_rate=hp.initial_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=n_iterations_per_epoch,
                                                   decay_rate=0.96,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-9)

        # Apply gradient clipping
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (grad, var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

    with tf.name_scope('Evaluation'):
        # Bass
        pred_b = tf.argmax(b_logits, axis=2, output_type=tf.int32)
        mask_b = tf.cast(tf.tile(tf.expand_dims(seq_mask, 2), [1, 1, hp.inner_b_classes]), tf.bool)
        pred_b_mask = tf.boolean_mask(tf.one_hot(pred_b, depth=hp.inner_b_classes), mask_b)
        y_b_mask = tf.boolean_mask(tf.one_hot(y_b, depth=hp.inner_b_classes), mask_b)
        _, _, F1_b = _F1_score(pred_b_mask, y_b_mask)

        # Pitch classes
        pred_p = tf.cast(tf.round(tf.sigmoid(p_logits)), tf.int32)
        mask_p = tf.cast(tf.tile(tf.expand_dims(seq_mask, 2), [1, 1, hp.inner_p_classes]), tf.bool)
        pred_p_mask = tf.boolean_mask(pred_p, mask_p)
        y_p_mask = tf.boolean_mask(y_p, mask_p)
        _, _, F1_p = _F1_score(pred_p_mask, y_p_mask)
    tf.summary.scalar('F1_bass', F1_b)
    tf.summary.scalar('F1_pitch_classes', F1_p)


    graph_location = 'model\\'
    train_writer = tf.summary.FileWriter(graph_location + '\\train')
    valid_writer = tf.summary.FileWriter(graph_location + '\\valid')
    print('Saving graph to: %s' % graph_location)
    merged = tf.summary.merge_all()
    train_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=1)

    # Train for a fixed number of iterations
    print('training the model...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        startTime = time.time() # start time of training
        best_valid_score = [0.0, 0.0]
        in_succession = 0
        best_epoch = 0
        for step in range(hp.training_steps):
            # Training
            if step == 0:
                indices = range(n_sequences)
                batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]

            if step / n_iterations_per_epoch > 5:
                # Shuffle training instances
                indices = random.sample(range(n_sequences), n_sequences)
                batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]


            batch = (X['train']['root'][batch_indices[step % len(batch_indices)]],
                     X['train']['triad_type'][batch_indices[step % len(batch_indices)]],
                     X['train']['duration'][batch_indices[step % len(batch_indices)]],
                     Y['train']['bass'][batch_indices[step % len(batch_indices)]],
                     Y['train']['treble'][batch_indices[step % len(batch_indices)]],
                     Y['train']['lens'][batch_indices[step % len(batch_indices)]])

            train_run_list = [train_op, merged, pred_b, pred_p, loss, ce_b, ce_p, L2_regularizer, F1_b, F1_p, seq_mask]
            train_feed_fict = {x_r: batch[0],
                               x_tr: batch[1],
                               x_d: batch[2],
                               y_b: batch[3],
                               y_p: batch[4],
                               y_len: batch[5],
                               dropout: hp.drop,
                               is_training: True,
                               global_step: step + 1}
            _, train_summary, train_b_pred, train_p_pred, train_loss, train_ce_b, train_ce_p, train_L2, train_F1_b, train_F1_p, mask = sess.run(train_run_list, feed_dict=train_feed_fict)

            # Display training log
            if step % n_iterations_per_epoch == 0:
                if step == 0:
                    print('*~ ce_b %.4f, ce_p %.4f, L2 %.4f ~*' % (train_ce_b, train_ce_p, train_L2))
                train_writer.add_summary(train_summary, step)
                print("------ step %d, epoch %d: train_loss %.4f (b %.4f, p %.4f, L2 %.4f), train_F1: (b %.4f, p %.4f) ------"
                      % (step, step//n_iterations_per_epoch, train_loss, train_ce_b, train_ce_p,train_L2, train_F1_b, train_F1_p))
                print('len'.ljust(4, ' '), batch[5][0])
                print('mask'.ljust(4, ' '), ''.join(['T'.rjust(4, ' ') if m else 'F'.rjust(4, ' ') for m in mask[0]]))
                print('x_r'.ljust(4, ' '), ''.join([root_list[b].rjust(4, ' ') for b in batch[0][0]]))
                print('x_tr'.ljust(4, ' '), ''.join([triad_type_list[b].rjust(4, ' ') for b in batch[1][0]]))
                print('x_d'.ljust(4, ' '), ''.join([str(round(b,1))[:3].rjust(4, ' ') for b in batch[2][0]]))
                print('y_b'.ljust(4, ' '), ''.join([root_list[b].rjust(4, ' ') for b in batch[3][0]]))
                print('p_b'.ljust(4, ' '), ''.join([root_list[b].rjust(4, ' ') for b in train_b_pred[0]]))
                print('y_p'.ljust(4, ' '), ''.join([str(b) for b in batch[4][0,:8]]))
                print('p_p'.ljust(4, ' '), ''.join([str(b) for b in train_p_pred[0,:8]]))

            # Validation
            if step % n_iterations_per_epoch == 0:
                valid_run_list = [merged, pred_b, pred_p, loss, ce_b, ce_p, L2_regularizer, F1_b, F1_p, seq_mask]
                valid_feed_fict = {x_r: X['test']['root'],
                                   x_tr: X['test']['triad_type'],
                                   x_d: X['test']['duration'],
                                   y_b: Y['test']['bass'],
                                   y_p: Y['test']['treble'],
                                   y_len: Y['test']['lens'],
                                   dropout: 0,
                                   is_training: False}
                valid_summary, valid_b_pred, valid_p_pred, valid_loss, valid_ce_b, valid_ce_p, valid_L2, valid_F1_b, valid_F1_p, mask = sess.run(valid_run_list, feed_dict=valid_feed_fict)
                valid_writer.add_summary(valid_summary, step)
                print("==== epoch %d: valid_loss %.4f (b %.4f, p %.4f, L2 %.4f), valid_F1: (b %.4f, p %.4f) ===="
                      % (step // n_iterations_per_epoch, valid_loss, valid_ce_b, valid_ce_p, valid_L2, valid_F1_b, valid_F1_p))
                sample_id = random.randint(0, X['test'].shape[0] - 1)
                print('len'.ljust(4, ' '), Y['test']['lens'][sample_id])
                print('mask'.ljust(4, ' '), ''.join(['T'.rjust(4, ' ') if m else 'F'.rjust(4, ' ') for m in mask[sample_id]]))
                print('x_r'.ljust(4, ' '), ''.join([root_list[b].rjust(4, ' ') for b in X['test']['root'][sample_id]]))
                print('x_tr'.ljust(4, ' '), ''.join([triad_type_list[b].rjust(4, ' ') for b in X['test']['triad_type'][sample_id]]))
                print('x_d'.ljust(4, ' '), ''.join([str(round(b, 1))[:3].rjust(4, ' ') for b in X['test']['duration'][sample_id]]))
                print('y_b'.ljust(4, ' '), ''.join([root_list[b].rjust(4, ' ') for b in Y['test']['bass'][sample_id]]))
                print('p_b'.ljust(4, ' '), ''.join([root_list[b].rjust(4, ' ') for b in valid_b_pred[sample_id]]))
                print('y_p'.ljust(4, ' '), ''.join([str(b) for b in Y['test']['treble'][sample_id, :8]]))
                print('p_p'.ljust(4, ' '), ''.join([str(b) for b in valid_p_pred[sample_id, :8]]))

                # Check if early stop
                if step > 0 and (valid_F1_b + valid_F1_p) / 2 > sum(best_valid_score) / 2:
                    best_valid_score = [valid_F1_b, valid_F1_p]
                    best_epoch = step // n_iterations_per_epoch
                    in_succession = 0
                    # Save parameters
                    print('*save patameters...')
                    saver.save(sess, graph_location + '\\coloring_model_' + str(hp.validation_set_id) + '.ckpt')
                else:
                    in_succession += 1
                    if in_succession > hp.n_in_succession:
                        print('Early stop.')
                        break

        elapsed_time = time.time() - startTime
        np.set_printoptions(precision=4)
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best valid_score =', np.round(best_valid_score, 4))
        print('best epoch = ', best_epoch)

def cross_validate_voicing_model(data_dir, hp):

    # Load training data
    print('Cross-validation: validation_set_id =', hp.validation_set_id)
    print('sequential_model =', hp.sequential_model)
    data = load_data(data_dir, hp.validation_set_id)
    X = {'train': {'bass': data['train']['bass'],
                  'treble': np.reshape([chroma for chroma in data['train']['treble'].flatten()], newshape=[data['train'].shape[0], data['train'].shape[1], 12]),
                  'duration': data['train']['duration']},
         'test':  {'bass': data['test']['bass'],
                  'treble': np.reshape([chroma for chroma in data['test']['treble'].flatten()], newshape=[data['test'].shape[0], data['test'].shape[1], 12]),
                  'duration': data['test']['duration']}}
    Y = {'train': {'piano_vector': np.reshape([vector for vector in data['train']['piano_vector'].flatten()], newshape=[data['train'].shape[0], data['train'].shape[1], 88]),
                  'lens': _get_valid_length(data['train']['duration'])},
         'test':  {'piano_vector': np.reshape([vector for vector in data['test']['piano_vector'].flatten()], newshape=[data['test'].shape[0], data['test'].shape[1], 88]),
                  'lens': _get_valid_length(data['test']['duration'])}}

    n_sequences = X['train']['bass'].shape[0]
    n_iterations_per_epoch = math.ceil(n_sequences / hp.batch_size)
    print('number of training sequences =', n_sequences)
    print('number of testing sequences =', X['test']['bass'].shape[0])
    print('n_iterations_per_epoch=', n_iterations_per_epoch)

    # Print hyperparameters
    print(hp)

    with tf.name_scope('Voicing_model'):
        # Placeholders
        x_b = tf.placeholder(tf.int32, [None, hp.n_steps], name="bass")
        x_p = tf.placeholder(tf.int32, [None, hp.n_steps, 12], name="pitch_calsses")
        x_d = tf.placeholder(tf.float32, [None, hp.n_steps], name="duration")
        y_v = tf.placeholder(tf.int32, [None, hp.n_steps, 88], name="voicing")
        y_len = tf.placeholder(tf.int32, [None], name="seq_lens")
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')

        if hp.sequential_model == 'blstm':
            v_logits, b_logits, p_logits, pianoroll_mask_float = BLSTM_voicing(x_b, x_p, x_d, y_len, dropout, hp)
        elif hp.sequential_model == 'mhsa':
            v_logits, b_logits, p_logits, pianoroll_mask_float = Attention_voicing(x_b, x_p, x_d, y_len, dropout, is_training, hp)
        else:
            print('Invalid model name.')
            exit(1)

    with tf.name_scope('Loss'):
        seq_mask = tf.sequence_mask(lengths=y_len, maxlen=tf.shape(x_b)[1], dtype=tf.float32) # seqence mask by which paddings are filtered, [batch, n_steps]

        # Voicing
        if hp.with_mask:
            ce_v = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_v, logits=v_logits, weights=pianoroll_mask_float, label_smoothing=0.0)
        else:
            ce_v = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_v, logits=v_logits, weights=seq_mask[:,:,tf.newaxis], label_smoothing=0.0)

        # Bass and pitch classes (in terms of chroma vector)
        ce_chroma = 0.5 * tf.losses.mean_squared_error(labels=tf.one_hot(x_b, depth=hp.inner_b_classes)[:,:,:12], predictions=tf.sigmoid(b_logits), weights=seq_mask[:, :, tf.newaxis]) \
                    + 0.5 * tf.losses.sigmoid_cross_entropy(multi_class_labels=x_p, logits=p_logits, weights=seq_mask[:, :, tf.newaxis])

        # L2 norm regularization
        vars = tf.trainable_variables()
        L2_regularizer = hp.voicing_beta_L2 * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])

        # Total loss
        loss = ce_v + ce_chroma + L2_regularizer
    tf.summary.scalar('Loss_total', loss)
    tf.summary.scalar('Loss_voicing', ce_v)
    tf.summary.scalar('Loss_chroma', ce_chroma)
    tf.summary.scalar('Loss_L2', L2_regularizer)

    with tf.name_scope('Optimization'):
        # apply learning rate decay
        learning_rate = tf.train.exponential_decay(learning_rate=hp.initial_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=n_iterations_per_epoch,
                                                   decay_rate=0.96,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-9)

        # Apply gradient clipping
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (grad, var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

    with tf.name_scope('Evaluation'):
        # Voicing
        pred_v = tf.round(tf.sigmoid(v_logits)) * pianoroll_mask_float
        pred_v = tf.cast(pred_v, tf.int32)
        mask_bool = tf.cast(pianoroll_mask_float, tf.bool)
        pred_v_mask = tf.boolean_mask(pred_v, mask_bool)
        y_v_mask = tf.boolean_mask(y_v, mask_bool)
        P_v, R_v, F1_v = _F1_score(pred_v_mask, y_v_mask)

        # Self similarity matrix
        pred_v_SSM = _compute_SSM(pred_v)
        y_v_SSM = _compute_SSM(y_v)
        pred_v_SSM_mask = _SSM_masking(pred_v_SSM, seq_mask=seq_mask)
        y_v_SSM_mask = _SSM_masking(y_v_SSM, seq_mask=seq_mask)
        SSM_score = 1.0 - tf.reduce_mean(tf.abs(pred_v_SSM_mask - y_v_SSM_mask))
    tf.summary.scalar('F1_voicing', F1_v)
    tf.summary.scalar('Precision_voicing', P_v)
    tf.summary.scalar('Recall_voicing', R_v)
    tf.summary.scalar('SSM_score', SSM_score)

    graph_location = 'model\\'
    train_writer = tf.summary.FileWriter(graph_location + '\\train')
    valid_writer = tf.summary.FileWriter(graph_location + '\\valid')
    print('Saving graph to: %s' % graph_location)
    merged = tf.summary.merge_all()
    train_writer.add_graph(tf.get_default_graph())
    valid_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=1)

    # Train for a fixed number of iterations
    print('training the model...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        startTime = time.time() # start time of training
        best_valid_score = 0.0
        in_succession = 0
        best_epoch = 0
        for step in range(hp.training_steps):
            # Training
            if step == 0:
                indices = range(n_sequences)
                batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]

            if step / n_iterations_per_epoch > 5:
                # Shuffle training instances
                indices = random.sample(range(n_sequences), n_sequences)
                batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]

            batch = (X['train']['bass'][batch_indices[step % len(batch_indices)]],
                     X['train']['treble'][batch_indices[step % len(batch_indices)]],
                     X['train']['duration'][batch_indices[step % len(batch_indices)]],
                     Y['train']['piano_vector'][batch_indices[step % len(batch_indices)]],
                     Y['train']['lens'][batch_indices[step % len(batch_indices)]])

            train_run_list = [train_op, merged, pred_v, pianoroll_mask_float, loss, ce_v, ce_chroma, L2_regularizer, P_v, R_v, F1_v, SSM_score]
            train_feed_fict = {x_b: batch[0],
                               x_p: batch[1],
                               x_d: batch[2],
                               y_v: batch[3],
                               y_len: batch[4],
                               dropout: hp.drop,
                               is_training: True,
                               global_step: step + 1}
            _, train_summary, train_v_pred, train_mask, train_loss, train_ce_v, train_ce_chroma, train_L2, train_P_v, train_R_v, train_F1_v, train_SSM_score = sess.run(train_run_list, feed_dict=train_feed_fict)

            # Display training log
            if step % n_iterations_per_epoch == 0:
                if step == 0:
                    print('*~ ce_v %.4f, ce_chroma %.4f, L2 %.4f ~*' % (train_ce_v, train_ce_chroma, train_L2))
                train_writer.add_summary(train_summary, step)
                print("------ step %d, epoch %d: train_loss %.4f (v %.4f, chroma %.4f, L2 %.4f), train_scores (P %.4f, R %.4f, F1 %.4f, SSM %.4f) ------"
                      % (step, step//n_iterations_per_epoch, train_loss, train_ce_v, train_ce_chroma, train_L2, train_P_v, train_R_v, train_F1_v, train_SSM_score))
                print('y_v'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v]) + ']' for v in batch[3][0,:2]]))
                print('p_v'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v]) + ']' for v in train_v_pred[0,:2]]))
                print('mask'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v.astype(np.int32)]) + ']' for v in train_mask[0, :2]]))


            # Validation
            if step % n_iterations_per_epoch == 0:
                valid_run_list = [merged, pred_v, pianoroll_mask_float, loss, ce_v, ce_chroma, L2_regularizer, P_v, R_v, F1_v, SSM_score]
                valid_feed_fict = {x_b: X['test']['bass'],
                                   x_p: X['test']['treble'],
                                   x_d: X['test']['duration'],
                                   y_v: Y['test']['piano_vector'],
                                   y_len: Y['test']['lens'],
                                   dropout: 0.0,
                                   is_training: False}
                valid_summary, valid_v_pred, valid_mask, valid_loss, valid_ce_v, valid_ce_chroma, valid_L2, valid_P_v, valid_R_v, valid_F1_v, valid_SSM_score = sess.run(valid_run_list, feed_dict=valid_feed_fict)
                valid_writer.add_summary(valid_summary, step)
                print("==== epoch %d: valid_loss %.4f (v %.4f, chroma %.4f, L2 %.4f), valid_scores (P %.4f, R %.4f, F1 %.4f, SSM %.4f) ===="
                      % (step // n_iterations_per_epoch, valid_loss, valid_ce_v, valid_ce_chroma, valid_L2, valid_P_v, valid_R_v, valid_F1_v, valid_SSM_score))
                sample_id = random.randint(0, X['test']['bass'].shape[0] - 1)
                print('y_v'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v]) + ']' for v in Y['test']['piano_vector'][sample_id, :2]]))
                print('p_v'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v]) + ']' for v in valid_v_pred[sample_id, :2]]))
                print('mask'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v.astype(np.int32)]) + ']' for v in valid_mask[sample_id, :2]]))

                # Check if early stop
                if valid_F1_v > best_valid_score:
                    best_valid_score = valid_F1_v
                    best_epoch = step // n_iterations_per_epoch
                    in_succession = 0
                    # Save variables of the model
                    print('*save parameters...')
                    saver.save(sess, graph_location + '\\voicing_model_' + str(hp.validation_set_id) + '.ckpt')
                else:
                    in_succession += 1
                    if in_succession > hp.n_in_succession:
                        print('Early Stop.')
                        break

        elapsed_time = time.time() - startTime
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best epoch = ', best_epoch)
        print('best valid score =', np.round(best_valid_score,4))

def cross_validate_end2end_chord_jazzification(data_dir, hp):
    # Load dataset
    print('Cross-validation: validation_set_id =', hp.validation_set_id)
    data = load_data(data_dir, hp.validation_set_id)

    X = {'train': data['train'][['root', 'triad_type', 'duration']],
         'test': data['test'][['root', 'triad_type', 'duration']]}
    Y = {'train': {'piano_vector': np.reshape([vector for vector in data['train']['piano_vector'].flatten()], newshape=[data['train'].shape[0], data['train'].shape[1], 88]),
                   'lens': _get_valid_length(data['train']['duration'])},
         'test': {'piano_vector': np.reshape([vector for vector in data['test']['piano_vector'].flatten()], newshape=[data['test'].shape[0], data['test'].shape[1], 88]),
                  'lens': _get_valid_length(data['test']['duration'])}}

    n_sequences = X['train']['root'].shape[0]
    n_iterations_per_epoch = math.ceil(n_sequences / hp.batch_size)
    print('number of training sequences =', n_sequences)
    print('number of testing sequences =', X['test']['root'].shape[0])
    print('n_iterations_per_epoch=', n_iterations_per_epoch)

    with tf.name_scope('Model'):
        # Placeholders
        x_r = tf.placeholder(tf.int32, [None, hp.n_steps], name="root")
        x_tr = tf.placeholder(tf.int32, [None, hp.n_steps], name="triad_type")
        x_d = tf.placeholder(tf.float32, [None, hp.n_steps], name="duration")
        y_v = tf.placeholder(tf.int32, [None, hp.n_steps, 88], name="voicing")
        y_len = tf.placeholder(tf.int32, [None], name="seq_lens")
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')

        v_logits = BLSTM_end2end(x_r, x_tr, x_d, y_len, dropout, hp)

    with tf.name_scope('Loss'):
        seq_mask = tf.sequence_mask(lengths=y_len, maxlen=tf.shape(x_r)[1], dtype=tf.float32) # seqence mask by which paddings are filtered, [batch, n_steps]
        # sigmoid cross entropy
        ce_v = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_v, logits=v_logits, weights=seq_mask[:, :, tf.newaxis], label_smoothing=0.0)

        # cosine distance
        v_act_norm = tf.nn.l2_normalize(tf.nn.sigmoid(v_logits), axis=2)
        y_v_norm = tf.nn.l2_normalize(tf.cast(y_v, tf.float32), axis=2)
        cos_v = tf.losses.cosine_distance(labels=y_v_norm, predictions=v_act_norm, axis=2, weights=seq_mask[:, :, tf.newaxis])

        # L2 norm regularization
        vars = tf.trainable_variables()
        L2_regularizer = hp.end2end_beta_L2 * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])

        # Total loss
        loss = ce_v + cos_v + L2_regularizer
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Loss_v', ce_v)
    tf.summary.scalar('Loss_L2', L2_regularizer)

    with tf.name_scope('Optimization'):
        # apply learning rate decay
        learning_rate = tf.train.exponential_decay(learning_rate=hp.initial_learning_rate,
                                                   global_step=global_step,
                                                   decay_steps=(n_iterations_per_epoch),
                                                   decay_rate=0.96,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-9)

        # Apply gradient clipping
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (grad, var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

    with tf.name_scope('Evaluation'):
        pred_v = tf.round(tf.sigmoid(v_logits)) * seq_mask[:,:,tf.newaxis]
        pred_v = tf.cast(pred_v, tf.int32)
        mask_bool = tf.cast(seq_mask, tf.bool)
        pred_v_mask = tf.boolean_mask(pred_v, mask_bool)
        y_v_mask = tf.boolean_mask(y_v, mask_bool)
        P_v, R_v, F1_v = _F1_score(pred_v_mask, y_v_mask)
    tf.summary.scalar('F1_voicing', F1_v)


    graph_location = 'model\\'
    print('Saving graph to: %s' % graph_location)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(graph_location + '\\train')
    valid_writer = tf.summary.FileWriter(graph_location + '\\valid')
    train_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=1)

    # Train for a fixed number of iterations
    print('training the model...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        startTime = time.time() # start time of training
        best_valid_score = 0.0
        in_succession = 0
        best_epoch = 0
        for step in range(hp.training_steps):
            # Training
            if step == 0:
                indices = range(n_sequences)
                batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]

            if step / n_iterations_per_epoch > 5:
                # Shuffle training instances
                indices = random.sample(range(n_sequences), n_sequences)
                batch_indices = [indices[x:x + hp.batch_size] for x in range(0, len(indices), hp.batch_size)]

            batch = (X['train']['root'][batch_indices[step % len(batch_indices)]],
                     X['train']['triad_type'][batch_indices[step % len(batch_indices)]],
                     X['train']['duration'][batch_indices[step % len(batch_indices)]],
                     Y['train']['piano_vector'][batch_indices[step % len(batch_indices)]],
                     Y['train']['lens'][batch_indices[step % len(batch_indices)]])

            train_run_list = [train_op, merged, pred_v, loss, ce_v, cos_v, L2_regularizer, P_v, R_v, F1_v]
            train_feed_fict = {x_r: batch[0],
                               x_tr: batch[1],
                               x_d: batch[2],
                               y_v: batch[3],
                               y_len: batch[4],
                               dropout: hp.drop,
                               global_step: step + 1}
            _, train_summary, train_v_pred, train_loss, train_ce_v, train_cos_v, train_L2, train_P_v, train_R_v, train_F1_v = sess.run(train_run_list, feed_dict=train_feed_fict)

            # Display training log
            if step % n_iterations_per_epoch == 0:
                if step == 0:
                    print('*~ ce_v %.4f, cos_v %.4f, L2 %.4f ~*' % (train_ce_v, train_cos_v, train_L2))
                train_writer.add_summary(train_summary, step)
                print("------ step %d, epoch %d: train_loss %.4f (ce %.4f, cos %.4f, L2 %.4f), train_scores (P %.4f, R %.4f, F1 %.4f) ------"
                      % (step, step//n_iterations_per_epoch, train_loss, train_ce_v, train_cos_v, train_L2, train_P_v, train_R_v, train_F1_v))
                print('y_v'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v]) + ']' for v in batch[3][0,:2]]))
                print('p_v'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v]) + ']' for v in train_v_pred[0,:2]]))

            # Validation
            if step % n_iterations_per_epoch == 0:
                valid_run_list = [merged, pred_v, loss, ce_v, cos_v, L2_regularizer, P_v, R_v, F1_v]
                valid_feed_fict = {x_r: X['test']['root'],
                                   x_tr: X['test']['triad_type'],
                                   x_d: X['test']['duration'],
                                   y_v: Y['test']['piano_vector'],
                                   y_len: Y['test']['lens'],
                                   dropout: 0.0}
                valid_summary, valid_v_pred, valid_loss, valid_ce_v, valid_cos_v, valid_L2, valid_P_v, valid_R_v, valid_F1_v = sess.run(valid_run_list, feed_dict=valid_feed_fict)
                valid_writer.add_summary(valid_summary, step)
                print("==== epoch %d: valid_loss %.4f (ce %.4f, cos %.4f, L2 %.4f), valid_scores (P %.4f, R %.4f, F1 %.4f) ===="
                      % (step // n_iterations_per_epoch, valid_loss, valid_ce_v, valid_cos_v, valid_L2, valid_P_v, valid_R_v, valid_F1_v))
                sample_id = random.randint(0, X['test']['root'].shape[0] - 1)
                print('y_v'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v]) + ']' for v in Y['test']['piano_vector'][sample_id, :2]]))
                print('p_v'.ljust(4, ' '), ''.join(['[' + ''.join([str(b) for b in v]) + ']' for v in valid_v_pred[sample_id, :2]]))

                # check if early stop
                if valid_F1_v > best_valid_score:
                    best_valid_score = valid_F1_v
                    best_epoch = step // n_iterations_per_epoch
                    in_succession = 0
                    # Save parameters
                    print('*save parameters...')
                    saver.save(sess, graph_location + '\\end2end_chord_jazzification_' + str(hp.validation_set_id) + '.ckpt')
                else:
                    in_succession += 1
                    if in_succession > hp.n_in_succession:
                        print('Early stop.')
                        break

        elapsed_time = time.time() - startTime
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best epoch = ', best_epoch)
        print('best valid score =', np.round(best_valid_score,4))

def _generate_midi_from_voicings(voicings, roots, durations, valid_len, output_dir, qpm=120):
    """voicings = [time_steps, 88]
            duartions = [time_steps]
            valid_len = int
            qpm = quarter notes per minutes """

    # Create a PrettyMIDI object
    midi = pm.PrettyMIDI()
    # Create an Instrument instance
    program = pm.instrument_name_to_program('Bright Acoustic Piano')
    instrument = pm.Instrument(program=program)
    onset = 0

    # Iterate over voicings
    for root, voicing, duration in zip(roots[:valid_len], voicings[:valid_len], durations[:valid_len]):
        new_duration = duration * (60 / qpm)
        if root != 'None':
            note_numbers = [i + 21 for i, x in enumerate(voicing) if x == 1]
            for number in note_numbers:
                # Create a Note instance, starting at 0s and ending at .5s
                note = pm.Note(velocity=100, pitch=number, start=onset, end=onset + new_duration)
                # Add it to the instrument
                instrument.notes.append(note)
        onset += new_duration

    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)
    # Write out the MIDI data
    midi.write(output_dir)

def chord_jazzification_inference(hp, threshold=0.6, user_input=False, random_sample=False):
    '''Generate jazzified chord sequences from the JAAH dataset using the chord jazzification model trained on the chord jazzificaion dataset.
        threshold used for binarizing the voicing probabilities'''

    def _get_user_input():
        valid_syntax = ['C', 'D', 'E', 'F', 'G', 'A', 'B', '#', 'b', ':', 'M', 'm', 'a', 'd', ' ']
        valid_roots = [r + a for a in ['', '#', 'b'] for r in valid_syntax[:7]]
        valid_qualities = valid_syntax[10:14]

        def _transform_input(user_input): # transform the input to the required format
            note_name_dict = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
            accidental_dict = {'': 0, '#': 1, 'b': -1}
            note_name_conversion_dict = {rk + ak: (rv + av) % 12 for rk, rv in note_name_dict.items() for ak, av in accidental_dict.items()}
            triad_type2int_dict = {'M': 0, 'm': 1, 'a': 2, 'd': 3}
            structured_input = [x.split(':') for x in user_input.split(' ')]
            structured_input = [(note_name_conversion_dict[x[0]], triad_type2int_dict[x[1]], 1.0) for x in structured_input]
            return np.array(structured_input, dtype=[('root_id', np.int32), ('triad_type_id', np.int32), ('duration', np.float32)])[np.newaxis, :]

        print('Please enter a sequence of triads to jazzify.')
        print('Valid syntax: roots = {C, D, E, F, G, A, B}, accidentals = {#, b}, qualities = {M, m, a, d}')
        print('Input example: C:M D:m G:M A:m E:m F:M D:m G:M Ab:M Bb:M C:M')

        # Get user input
        while(True):
            user_input = input("Enter your sequence of triads: ")
            # Check input validity
            user_input_roots = [s.split(':')[0] for s in user_input.split(' ')]
            user_input_qualities = [s.split(':')[1] for s in user_input.split(' ')]
            if any(s not in valid_syntax for s in user_input):
                print('Invalid syntax')
                print([s not in valid_syntax for s in user_input])
            if any([s not in valid_roots for s in user_input_roots]):
                print('Invalid root.')
            if any([s not in valid_qualities for s in user_input_qualities]):
                print('Invalid quality.')
            else:
                break
        # Transform input to the required format
        structured_input = _transform_input(user_input)
        return structured_input


    if not user_input:
        print("Load inference data...")
        data_dir = 'JAAH_data.pickle' # the JAAH dataset

        with open(data_dir, 'rb') as file:
            data = pickle.load(file) # [('root', object), ('triad_type', object), ('color', object), ('bass', object), ('root_id', np.int32), ('triad_type_id', np.int32), ('bass_id', np.int32), ('duration', np.float32)]

        Y_len = _get_valid_length(data['duration'])
        print('JAAH data.shape =', data.shape)
        # print('data.dtype =', data.dtype)
        # print('Y_len.shape =', Y_len.shape)
        n_steps = data.shape[1]
    else:
        data = _get_user_input()
        n_steps = data.shape[1]
        Y_len = [n_steps]


    with tf.Graph().as_default() as g_coloring:
        with tf.name_scope('Chord_jazzification_coloring'):
            tf.get_default_graph()
            # Placeholders
            x_r = tf.placeholder(tf.int32, [None, n_steps], name="root")
            x_tr = tf.placeholder(tf.int32, [None, n_steps], name="triad_type")
            x_d = tf.placeholder(tf.float32, [None, n_steps], name="duration")
            y_len = tf.placeholder(tf.int32, [None], name="seq_lens")
            dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
            is_training = tf.placeholder(dtype=tf.bool, name='is_training')

            if hp.sequential_model == 'blstm':
                b_logits, p_logits = BLSTM_coloring(x_r, x_tr, x_d, y_len, dropout, hp)
            elif hp.sequential_model == 'mhsa':
                b_logits, p_logits = Attention_coloring(x_r, x_tr, x_d, y_len, dropout, is_training, hp)
            else:
                print('Invalid model name.')
                exit(1)

            pred_b = tf.argmax(b_logits, axis=2, output_type=tf.int32)
            pred_p = tf.cast(tf.round(tf.sigmoid(p_logits)), tf.int32)

        with tf.Session(graph=g_coloring) as coloring_sess:
            coloring_saver = tf.train.Saver(max_to_keep=1)
            model_dir = 'coloring_model\\coloring_model.ckpt'
            coloring_saver.restore(coloring_sess, model_dir)
            coloring_run_list = [pred_b, pred_p]
            coloring_feed_fict = {x_r: data['root_id'],
                                  x_tr: data['triad_type_id'],
                                  x_d: data['duration'],
                                  y_len: Y_len,
                                  dropout: 0,
                                  is_training: False}
            p_b, p_p = coloring_sess.run(coloring_run_list, feed_dict=coloring_feed_fict)

    with tf.Graph().as_default() as g_voicing:
        with tf.name_scope('Chord_jazzification_voicing'):
            # Placeholders
            x_b = tf.placeholder(tf.int32, [None, n_steps], name="bass")
            x_p = tf.placeholder(tf.int32, [None, n_steps, 12], name="pitch_calsses")
            x_d = tf.placeholder(tf.float32, [None, n_steps], name="duration")
            y_len = tf.placeholder(tf.int32, [None], name="seq_lens")
            dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
            is_training = tf.placeholder(dtype=tf.bool, name="is_training")

            if hp.sequential_model == 'blstm':
                v_logits, b_logits, p_logits, pianoroll_mask_float = BLSTM_voicing(x_b, x_p, x_d, y_len, dropout, hp)
            elif hp.sequential_model == 'mhsa':
                v_logits, b_logits, p_logits, pianoroll_mask_float = Attention_voicing(x_b, x_p, x_d, y_len, dropout, is_training, hp)
            else:
                print('Invalid model name.')
                exit(1)

            v_probs = (tf.sigmoid(v_logits) + (0.5 - threshold)) * pianoroll_mask_float
            if not random_sample:
                pred_v = tf.cast(tf.round(v_probs), tf.int32)
            else:
                pred_v = tf.ceil(v_probs - tf.random_uniform(tf.shape(v_probs)))

        with tf.Session(graph=g_voicing) as voicing_sess:
            voicing_saver = tf.train.Saver(max_to_keep=1)
            model_dir = 'voicing_model\\voicing_model.ckpt'
            voicing_saver.restore(voicing_sess, model_dir)

            voicing_feed_fict = {x_b: p_b,
                                 x_p: p_p,
                                 x_d: data['duration'],
                                 y_len: Y_len,
                                 dropout: 0,
                                 is_training: False}

            p_v, v_mask = voicing_sess.run([pred_v, pianoroll_mask_float], feed_dict=voicing_feed_fict)


    if not user_input:
        # Get the chord labels of JAAH sequences
        def _get_chord_sequences(data, Y_len):
            def get_chord_string(label):
                if label['root'] != 'None':
                    string = label['root'] + ':' + label['triad_type']
                    if label['color'] != 'None':
                        string += label['color']
                else:
                    string = 'None'
                if label['root'] != label['bass']:
                    string += ('/' + label['bass'])

                string += ('-' + str(label['duration']))
                return string

            chord_sequences = []
            for i, sequence in enumerate(data):
                chord_sequences.append([get_chord_string(l) for l in sequence[:Y_len[i]]])
            return chord_sequences

        JAAH_sequences = _get_chord_sequences(data, Y_len)
        # print(JAAH_sequences)

        # Compare generated voicings with input chord symbols
        def _get_colorings(data, voicing_sequences, Y_len):
            coloring_dict = {0: '1', 1: 'b2', 2: '2', 3: 'b3', 4: '3', 5: '4', 6: '#4', 7: '5', 8: 'b6', 9: '6', 10: 'm7', 11: 'M7'}
            def compare_chroma(label, v):
                if label['root'] != 'None':
                    label_chroma = np.roll(triad_id_to_chroma[label['triad_type_id']], shift=label['root_id'])
                    v_note_numbers = [i+21 for i, x in enumerate(v) if x==1]
                    v_chroma = np.array([1 if i in [number % 12 for number in v_note_numbers] else 0 for i in range(12)])
                    chroma_diff = v_chroma - label_chroma
                    if not all(x==0  for x in chroma_diff):
                        chroma_diff_shift = np.roll(chroma_diff, shift=-label['root_id'])
                        color = [coloring_dict[i] if x > 0 else 'o' + coloring_dict[i] for i, x in enumerate(chroma_diff_shift) if x != 0]
                        color = '(' + ','.join(color) + ')'
                    else:
                        color = 'None'

                    if sum(v) > 0:
                        v_bass_id = (np.where(v == 1)[0][0] + 21) % 12
                        if v_bass_id != label['root_id']:
                            v_bass = (v_bass_id - label['root_id']) % 12
                            v_bass = coloring_dict[v_bass]
                            color += ('/' + v_bass)

                else:
                    color = 'None'

                return color

            output_colorings = []
            for i, (labels, v_seq) in enumerate(zip(data, voicing_sequences)):
                output_colorings.append([compare_chroma(l, v) for l, v in zip(labels[:Y_len[i]], v_seq[:Y_len[i]])])
            return output_colorings

        model_colorings = _get_colorings(data, p_v, Y_len)

        # Generate midi example
        print('Generate midi...')
        generated_labels = []
        output_folder = 'JAAH_inference\\'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        for sample_id in range(p_v.shape[0]):
            output_info = [l + ', ' + c for l, c in zip(JAAH_sequences[sample_id], model_colorings[sample_id])]
            # print(str(sample_id+1) + ':', output_info)
            generated_labels.append(str(sample_id+1) + ':' + str(output_info) + '\n')
            output_dir = output_folder + str(sample_id+1) + '.mid'
            _generate_midi_from_voicings(p_v[sample_id], data['root_id'][sample_id], data['duration'][sample_id], Y_len[sample_id], output_dir=output_dir)

        with open(output_folder + "chord_sequences.txt", "w") as file:
            file.writelines(generated_labels)


    else:
        output_folder = 'user_inference\\'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        for sample_id in range(p_v.shape[0]):
            output_dir = output_folder + str(sample_id+1) + '.mid'
            _generate_midi_from_voicings(p_v[sample_id], data['root_id'][sample_id], data['duration'][sample_id], Y_len[sample_id], output_dir=output_dir, qpm=60)

    print('Chord jazzification completed.')





