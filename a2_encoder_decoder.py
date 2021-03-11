"""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall

All of the files in this directory and all subdirectories are:
Copyright (c) 2021 University of Toronto
"""

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase


# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):

    def init_submodules(self):
        # Hints:
        # 1. You must initialize these submodules:
        #   self.rnn, self.embedding
        # 2. You will need these object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type,
        #   self.hidden_state_size, self.num_hidden_layers.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, GRU, RNN, Embedding}
        args = [self.word_embedding_size, self.hidden_state_size]
        kwargs = {'num_layers': self.num_hidden_layers,
                  'dropout': self.dropout,
                  'bidirectional': True}

        if self.cell_type == "lstm":
            self.rnn = torch.nn.LSTM(*args, **kwargs)
        elif self.cell_type == "gru":
            self.rnn = torch.nn.GRU(*args, **kwargs)
        elif self.cell_type == "rnn":
            self.rnn = torch.nn.RNN(*args, **kwargs)
        else:
            raise NameError("cell_type not defined")

        self.embedding = torch.nn.Embedding(self.source_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)

    def forward_pass(self, F, F_lens, h_pad=0.):
        # Recall:
        #   F is size (S, M)
        #   F_lens is of size (M,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use these methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states
        x = self.get_all_rnn_inputs(F)
        h = self.get_all_hidden_states(x, F_lens, h_pad)

        return h

    def get_all_rnn_inputs(self, F):
        # Recall:
        #   F is size (S, M)
        #   x (output) is size (S, M, I)
        x = self.embedding(F)

        return x

    def get_all_hidden_states(self, x, F_lens, h_pad):
        # Recall:
        #   x is of size (S, M, I)
        #   F_lens is of size (M,)
        #   h_pad is a float
        #   h (output) is of size (S, M, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        # x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens,
        #                                                    enforce_sorted=False)
        # # h_packed, h_n = self.rnn(x_packed)
        # # h, len_unpacked = torch.nn.utils.rnn.pad_packed_sequence(h_packed)
        # h_packed, _ = self.rnn.forward(x_packed)
        # h, _ = torch.nn.utils.rnn.pad_packed_sequence(h_packed,
        #                                               padding_value=h_pad)
        # return h
        x = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens,
                                                    enforce_sorted=False)
        outputs, _ = self.rnn.forward(x)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                                                            padding_value=h_pad)
        return outputs


class DecoderWithoutAttention(DecoderBase):
    """A recurrent decoder without attention"""

    def init_submodules(self):
        # Hints:
        # 1. You must initialize these submodules:
        #   self.embedding, self.cell, self.ff
        # 2. You will need these object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        if self.cell_type == "lstm":
            self.cell = torch.nn.LSTMCell(self.word_embedding_size,
                                          self.hidden_state_size)
        elif self.cell_type == "gru":
            self.cell = torch.nn.GRUCell(self.word_embedding_size,
                                         self.hidden_state_size)
        elif self.cell_type == "rnn":
            self.cell = torch.nn.RNNCell(self.word_embedding_size,
                                         self.hidden_state_size)
        else:
            raise NameError("cell_type not defined")

        self.embedding = torch.nn.Embedding(self.target_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)

        # or self.hidden_state_size * 2 ???
        self.ff = torch.nn.Linear(self.hidden_state_size,
                                  self.target_vocab_size)

    def forward_pass(self, E_tm1, htilde_tm1, h, F_lens):
        # Recall:
        #   E_tm1 is of size (M,)
        #   htilde_tm1 is of size (M, 2 * H)
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   logits_t (output) is of size (M, V)
        #   htilde_t (output) is of same size as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use these methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.
        xtilde_t = self.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)
        htilde_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)
        logits_t = self.get_current_logits(htilde_t)

        return logits_t, htilde_t

    def get_first_hidden_state(self, h, F_lens):
        # Recall:
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   htilde_tm1 (output) is of size (M, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch functions: torch.cat
        hidden_size = h.shape[2]
        m = F_lens.shape[0]
        forward_direction = h[:, :, 0:hidden_size // 2]
        backward_direction = h[:, :, hidden_size // 2:]

        forward_extract = \
            [forward_direction[F_lens[i], i].reshape(1, hidden_size)
             for i in range(m)]
        # tensors of shape [m, hidden_size]
        forward_hidden = torch.cat(forward_extract, 0)
        backward_hidden = backward_direction[0]
        htilde_0 = torch.cat((forward_hidden, backward_hidden), 1)

        return htilde_0

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):

        # Recall:
        #   E_tm1 is of size (M,)
        #   htilde_tm1 is of size (M, 2 * H) or a tuple of two of those (LSTM)
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   xtilde_t (output) is of size (M, Itilde)
        xtilde_t = self.embedding(E_tm1)

        return xtilde_t

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        # Recall:
        #   xtilde_t is of size (M, Itilde)
        #   htilde_tm1 is of size (M, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same size as htilde_tm1
        htilde_t = self.cell(xtilde_t, htilde_tm1)

        return htilde_t

    def get_current_logits(self, htilde_t):
        # Recall:
        #   htilde_t is of size (M, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of size (M, V)
        logits_t = self.ff(htilde_t)

        return logits_t


class DecoderWithAttention(DecoderWithoutAttention):
    """A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    """

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.ff
        # 2. You will need these object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.
        if self.cell_type == "lstm":
            self.cell = torch.nn.LSTMCell(self.word_embedding_size +
                                          self.hidden_state_size,
                                          self.hidden_state_size)
        elif self.cell_type == "gru":
            self.cell = torch.nn.GRUCell(self.word_embedding_size +
                                         self.hidden_state_size,
                                         self.hidden_state_size)
        elif self.cell_type == "rnn":
            self.cell = torch.nn.RNNCell(self.word_embedding_size +
                                         self.hidden_state_size,
                                         self.hidden_state_size)
        else:
            raise NameError("cell_type not defined")

        self.embedding = torch.nn.Embedding(self.target_vocab_size,
                                            self.word_embedding_size,
                                            padding_idx=self.pad_id)

        self.ff = torch.nn.Linear(self.hidden_state_size,
                                  self.target_vocab_size)

    def get_first_hidden_state(self, h, F_lens):
        # Hint: For this time, the hidden states should be initialized to zeros.
        hidden_size = h.shape[2]
        m = h.shape[1]
        htilde_0 = torch.zeros([m, hidden_size], device=h.device)

        return htilde_0

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h, F_lens):
        # Hint: Use attend() for c_t
        embed_tm1 = self.embedding(E_tm1)  # (M, word_embedding_size)
        c_tm1 = self.attend(htilde_tm1, h, F_lens)  # (M, hidden_state_size)
        xtilde_t = torch.cat([embed_tm1, c_tm1], 1)

        return xtilde_t

    def attend(self, htilde_t, h, F_lens):
        """The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of size ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of size ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of size ``(M, self.hidden_state_size)``. The
            context vectorc_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        """
        alpha_t = self.get_attention_weights(htilde_t, h, F_lens)
        hidden_size = h.shape[2]
        # alpha_tile = alpha_t.repeat(1, 1, hidden_size)
        alpha_tile = torch.cat([alpha_t.unsqueeze(2)
                                for i in range(hidden_size)], dim=2)
        c_s = alpha_tile * h
        c_t = torch.sum(c_s, dim=0)

        return c_t

    def get_attention_weights(self, htilde_t, h, F_lens):
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_energy_scores()
        # alpha_t (output) is of size (S, M)
        e_t = self.get_energy_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens  # (S, M)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_energy_scores(self, htilde_t, h):
        # Recall:
        #   htilde_t is of size (M, 2 * H)
        #   h is of size (S, M, 2 * H)
        #   e_t (output) is of size (S, M)
        #
        # Hint:
        # Relevant pytorch functions: torch.nn.functional.cosine_similarity
        s = h.shape[0]
        htilde_tile = htilde_t.repeat(s, 1, 1)
        e_t = torch.nn.functional.cosine_similarity(h, htilde_tile, 2)

        return e_t


class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not modify this line

        # Hints:
        # 1. The above line should ensure self.ff, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize these submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need these object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once.
        #    They should not be lists!
        # 5. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)
        # 6. You do *NOT* need self.heads at this point
        self.W = torch.nn.Linear(self.hidden_state_size,
                                 self.hidden_state_size, bias=False)
        self.Wtilde = torch.nn.Linear(self.hidden_state_size,
                                      self.hidden_state_size, bias=False)
        self.Q = torch.nn.Linear(self.hidden_state_size,
                                 self.hidden_state_size, bias=False)

    def attend(self, htilde_t, h, F_lens):
        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch functions:
        #   tensor().repeat_interleave, tensor().view
        # 3. You *WILL* need self.heads at this point
        # 4. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.

        # h: (S, M, hidden_size), htilde_t: (M, hidden_size)
        # split by N heads
        # call super().attend()
        # combine c_t
        n = self.heads
        s, m, hidden_size = h.shape[0], h.shape[1], h.shape[2]
        htilde_n = self.Wtilde(htilde_t).repeat_interleave(n).view(m, -1)
        h_n = self.W(h).repeat_interleave(n).view(s, m, -1)
        c_n = super().attend(htilde_n, h_n, F_lens)  # (M, N * hidden_size)
        c_t = c_n.view(m, -1, n, 1)[:, :, 0, 0]  # (M, hidden_size)
        c_combine = self.Q(c_t)

        return c_combine


class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(self, encoder_class, decoder_class):
        # Hints:
        # 1. You must initialize these submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need these object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos, self.heads
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it
        self.encoder = encoder_class(self.source_vocab_size, self.source_pad_id,
                                     self.word_embedding_size,
                                     self.encoder_num_hidden_layers,
                                     self.encoder_hidden_size,
                                     self.encoder_dropout, self.cell_type)
        self.encoder.init_submodules()
        self.decoder = decoder_class(self.target_vocab_size, self.target_eos,
                                     self.word_embedding_size,
                                     self.encoder_hidden_size * 2,
                                     self.cell_type)
        self.decoder.init_submodules()

    def get_logits_for_teacher_forcing(self, h, F_lens, E):
        # Recall:
        #   h is of size (S, M, 2 * H)
        #   F_lens is of size (M,)
        #   E is of size (T, M)
        #   logits (output) is of size (T - 1, M, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than E (why?)
        logits = []
        htilde_tml = None

        if htilde_tml is None:
            htilde_tml = self.decoder.get_first_hidden_state(h, F_lens)
            if self.cell_type == "lstm":  # initialize LSTM cell state to zero
                htilde_tml = (htilde_tml, torch.zeros_like(htilde_tml))

        for i in range(E.shape[0]):
            logits_t, htilde_tm1 = \
                self.decoder.forward_pass(E[i], htilde_tml, h, F_lens)
            logits.append(logits_t)

        logits = torch.stack(logits[1:], dim=0)  # (T - 1, M, V)

        return logits

    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of size (M, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of size (M, K)
        #   b_tm1_1 is of size (t, M, K)
        #   b_t_0 (first output) is of size (M, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (second output) is of size (t + 1, M, K)
        #   logpb_t (third output) is of size (M, K)
        #
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of size z of (A, B),
        #   then the element z[a, b] maps to z'[a*B + b]
        K = self.beam_width
        V = logpy_t.shape[2]
        extensions_t = logpb_tm1.unsqueeze(-1) + logpy_t  # (M, K, V)
        extensions_t = torch.flatten(extensions_t, 1)  # (M, K * V)

        logpb_t, v = torch.topk(extensions_t, K, dim=1)  # (M, K)
        path = (v // V).unsqueeze(-1)  # (M, K) -> (M, K, 1)
        v = (v % V).unsqueeze(0)  # (M, K) -> (1, M, K)
        b_tm1_1 = torch.gather(b_tm1_1, 2,
                               path.squeeze(-1).unsqueeze(0).expand_as(b_tm1_1))
        b_t_1 = torch.cat([b_tm1_1, v], dim=0)  # (t + 1, M, K)

        if self.cell_type == "lstm":
            b_t_0 = (torch.gather(htilde_t[0], 1, path.expand_as(htilde_t[0])),
                     torch.gather(htilde_t[1], 1, path.expand_as(htilde_t[1])))
        else:
            b_t_0 = torch.gather(htilde_t, 1, path.expand_as(htilde_t))

        return b_t_0, b_t_1, logpb_t
