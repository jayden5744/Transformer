import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Embedding(pl.LightningModule):
    def __init__(self, d_input: int, d_hidden: int):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(d_input, d_hidden).to(device)
        self.d_embedding = d_hidden

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(pl.LightningModule):
    def __init__(self, d_hidden: int, max_seq_len: int = 50):
        super(PositionalEncoding, self).__init__()
        self.d_embedding = d_hidden
        self.max_seq_len = max_seq_len
        # todo: Debug 1
        # max_seq_len -> max_seq_len + 1
        sinusoid_table = np.array([self.get_position_angle_vec(pos_i) for pos_i in range(max_seq_len + 1)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(sinusoid_table), freeze=True).to(device)

    def forward(self, x, pad_id: int):
        positions = torch.arange(x.size(1), device=x.device, dtype=x.dtype).expand(x.size(0),
                                                                                   x.size(1)).contiguous() + 1
        pos_mask = x.eq(pad_id)  # padding을 masking
        positions.masked_fill_(pos_mask, 0).to(device)  # True는 0으로 masking
        return self.embedding(positions)

    def cal_angle(self, position, i):
        return position / np.power(10000, 2 * (i // 2) / self.d_embedding)

    def get_position_angle_vec(self, position):
        return [self.cal_angle(position, hid_j) for hid_j in range(self.d_embedding)]


class ScaledDotProductAttention(pl.LightningModule):
    def __init__(self, d_head):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = math.sqrt(d_head)

    def forward(self, q, k, v, mask):
        """
        Args:
            q:
            k:
            v:
            mask:
        Returns:
        """
        # Q x K^T, attention_score's shape: (n_batch, len_q, len_k)
        attention_score = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        if mask is not None:  # masking
            attention_score = attention_score.masked_fill(mask, -1e9)

        # softmax, attention_prob's shape: (n_batch, len_q, len_k)
        attention_prob = F.softmax(attention_score, dim=-1)

        # Attention_Prob x V, out's shape: (n_batch, len_q, len_k)
        out = torch.matmul(attention_prob, v)
        return out, attention_prob


class MultiHeadAttention(pl.LightningModule):
    def __init__(self, d_hidden: int, n_heads: int):
        """
        Args:
            d_hidden:
            n_heads:
        """
        super(MultiHeadAttention, self).__init__()

        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads
        self.norm_layer = nn.LayerNorm(d_hidden, eps=1e-6)
        self.attention = ScaledDotProductAttention(self.d_head)

        self.q_weight = nn.Linear(d_hidden, d_hidden).to(device)
        self.k_weight = nn.Linear(d_hidden, d_hidden).to(device)
        self.v_weight = nn.Linear(d_hidden, d_hidden).to(device)

        self.fc_layer = nn.Linear(d_hidden, d_hidden).to(device)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q:
            k:
            v:
            mask:
        Returns:
        """
        residual, n_batch = q, q.size(0)

        # q_s, k_s, v_s => [batch_size, n_heads, len_q, d_k]
        q_s = self.q_weight(q).view(n_batch, -1, self.n_heads, self.d_head).transpose(1, 2)
        k_s = self.k_weight(k).view(n_batch, -1, self.n_heads, self.d_head).transpose(1, 2)
        v_s = self.v_weight(v).view(n_batch, -1, self.n_heads, self.d_head).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1).to(device)  # => [n_batch, n_head, len_q, len_k]

        # context: [n_batch, n_head, len_q, len_k]
        # attention_prob: [n_batch, len_q, len_k]
        context, attention_prob = self.attention(q_s, k_s, v_s, mask)

        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(n_batch, -1, self.d_hidden)

        output = self.fc_layer(context)
        top_attn = attention_prob.view(n_batch, self.n_heads, q.size(1), k.size(1))[:, 0, :, :].contiguous()
        return output, top_attn


class PositionWiseFeedForwardLayer(pl.LightningModule):
    """
    Position Wise FFNN(x) = MAX(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_hidden, d_pf, activation: str = "Relu"):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.d_hidden = d_hidden
        self.d_pf = d_pf

        self.layer_1 = nn.Linear(d_hidden, d_pf).to(device)
        self.layer_2 = nn.Linear(d_pf, d_hidden).to(device)

        if activation.lower() == "relu":
            self.activate = F.relu
        elif activation.lower() == "gelu":
            self.activate = F.gelu
        else:
            raise Exception

    def forward(self, inputs):
        """
        Args:
            inputs:
        Returns:
        """
        out = self.layer_1(inputs)
        out = self.activate(out).to(device)
        out = self.layer_2(out)
        return out


class EncoderLayer(pl.LightningModule):
    def __init__(self, d_hidden, n_heads, d_ff, dropout_rate=0.0, layer_norm_epsilon=1e-12):
        super(EncoderLayer, self).__init__()
        self.enc_self_attention = MultiHeadAttention(d_hidden, n_heads)
        self.pwff = PositionWiseFeedForwardLayer(d_hidden, d_ff)

        self.layer_norm1 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon).to(device)
        self.residual_dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon).to(device)
        self.residual_dropout2 = nn.Dropout(dropout_rate)

    def forward(self, enc_inputs, enc_mask):
        # enc_inputs to same q, k, v
        # Multi-head Self- Attention
        attention_output, attention_prob = self.enc_self_attention(enc_inputs, enc_inputs, enc_inputs, enc_mask)
        # residual Dropout
        attention_output = self.residual_dropout1(attention_output)
        norm_attention_output = self.layer_norm1(enc_inputs + attention_output)

        # Position-wise FFNN
        pwff_outputs = self.pwff(norm_attention_output)
        # residual Dropout
        pwff_outputs = self.residual_dropout2(pwff_outputs)
        norm_pwff_outputs = self.layer_norm2(pwff_outputs + norm_attention_output)
        return norm_pwff_outputs, attention_prob


class Encoder(pl.LightningModule):
    def __init__(self, d_input, d_hidden, n_layers, n_heads, d_ff, dropout_rate: float = 0.0, max_length=50, padding_id=3):
        super(Encoder, self).__init__()
        self.src_emb = Embedding(d_input, d_hidden)
        self.pos_emb = PositionalEncoding(d_hidden, max_length)
        self.encoder_dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([EncoderLayer(d_hidden, n_heads, d_ff, dropout_rate)
                                     for _ in range(n_layers)])
        self.pad_id = padding_id

    def forward(self, enc_inputs):

        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(enc_inputs, self.pad_id)  # => [bs, max_seq_len, d_hidden]
        # encoder dropout
        enc_outputs = self.encoder_dropout(enc_outputs)

        enc_self_attn_mask = self.get_attn_pad_mask(enc_inputs, enc_inputs)  # => [bs, max_seq_len, max_seq_len]

        enc_self_attentions = []
        for layer in self.layers:
            # enc_outputs => [bs, max_seq_len, d_hidden]
            enc_outputs, enc_self_attention = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attentions.append(enc_self_attention)

        return enc_outputs, enc_self_attentions

    def get_attn_pad_mask(self, seq_q, seq_k):
        # seq_q, seq_k => [bs, seq_len] 입력문장
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(self.pad_id).unsqueeze(1)  # => [bs, 1, seq_q(=seq_k)]
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # => [bs, len_q, len_k]


class DecoderLayer(pl.LightningModule):
    def __init__(self, d_hidden, n_heads, d_ff, dropout_rate: float = 0.0, layer_norm_epsilon=1e-12):
        super(DecoderLayer, self).__init__()
        self.dec_self_attention = MultiHeadAttention(d_hidden, n_heads)
        self.dec_enc_attention = MultiHeadAttention(d_hidden, n_heads)
        self.pwff = PositionWiseFeedForwardLayer(d_hidden, d_ff)

        self.residual_dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon).to(device)
        self.residual_dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon).to(device)
        self.residual_dropout3 = nn.Dropout(dropout_rate)
        self.layer_norm3 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon).to(device)

    def forward(self, dec_inputs, enc_outputs, dec_mask, dec_enc_mask):
        # Masked Multi-head Self- Attention
        masked_outputs, masked_attention_prob = self.dec_self_attention(dec_inputs, dec_inputs, dec_inputs,
                                                                        dec_mask)
        # Residual Dropout
        masked_outputs = self.residual_dropout1(masked_outputs)

        # Layer Normalization
        norm_masked_outputs = self.layer_norm1(dec_inputs + masked_outputs)

        # Multi-head Self- Attention
        dec_enc_outputs, dec_enc_attention_prob = self.dec_enc_attention(norm_masked_outputs, enc_outputs,
                                                                         enc_outputs, dec_enc_mask)
        # Residual Dropout
        dec_enc_outputs = self.residual_dropout2(dec_enc_outputs)

        # Layer Normalization
        norm_dec_enc_outputs = self.layer_norm2(norm_masked_outputs + dec_enc_outputs)

        # Position-wise FFNN
        ffnn_outputs = self.pwff(norm_dec_enc_outputs)

        # Residual Dropout
        ffnn_outputs = self.residual_dropout2(ffnn_outputs)

        # Layer Normalization
        norm_ffnn_outputs = self.layer_norm3(norm_dec_enc_outputs + ffnn_outputs)

        return norm_ffnn_outputs, masked_attention_prob, dec_enc_attention_prob


class Decoder(pl.LightningModule):
    def __init__(self, d_input, d_hidden, n_layers, n_heads, d_ff, dropout_rate=0.0, max_length=50, padding_id=3):
        super(Decoder, self).__init__()
        self.trg_emb = Embedding(d_input, d_hidden)
        self.pos_emb = PositionalEncoding(d_hidden, max_length)
        self.decoder_dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([DecoderLayer(d_hidden, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])
        self.pad_id = padding_id

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.trg_emb(dec_inputs) + self.pos_emb(dec_inputs, self.pad_id)
        # decoder dropout
        dec_outputs = self.decoder_dropout(dec_outputs)
        input_mask = self.get_dec_input_mask(dec_inputs)
        attention_mask = self.get_attn_pad_mask(dec_inputs, enc_inputs)

        for layer in self.layers:
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, input_mask,
                                                                   attention_mask)
        return dec_outputs

    def get_dec_input_mask(self, x):
        attention_mask = self.get_attn_pad_mask(x, x)

        subsequent_mask = torch.ones_like(x).unsqueeze(-1).expand(x.size(0), x.size(1), x.size(1))
        subsequent_mask = subsequent_mask.triu(diagonal=1)  # upper triangular part of a matrix(2-D)

        return torch.gt((attention_mask + subsequent_mask), 0)

    def get_attn_pad_mask(self, seq_q, seq_k):
        # seq_q, seq_k => [bs, seq_len] 입력문장
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(self.pad_id).unsqueeze(1)  # => [bs, 1, seq_q(=seq_k)]
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # => [bs, len_q, len_k]


class Transformer(pl.LightningModule):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        hp = cfg.model
        self.encoder = Encoder(d_input=hp.enc_vocab_size, d_hidden=hp.enc_hidden_dim, n_layers=hp.enc_layers,
                               n_heads=hp.enc_heads, d_ff=hp.enc_ff_dim, dropout_rate=hp.dropout_rate,
                               max_length=hp.max_sequence_len)
        self.decoder = Decoder(d_input=hp.dec_vocab_size, d_hidden=hp.dec_hidden_dim, n_layers=hp.dec_layers,
                               n_heads=hp.dec_heads, d_ff=hp.dec_ff_dim, dropout_rate=hp.dropout_rate,
                               max_length=hp.max_sequence_len)
        self.linear = nn.Linear(hp.dec_hidden_dim, hp.dec_vocab_size).to(device)

    def forward(self, enc_inputs, dec_inputs, visualize=False):
        # enc_inputs => [bs, max_seq_len]
        # dec_inputs => [bs, max_seq_len]
        enc_outputs, enc_attentions = self.encode(enc_inputs)
        outputs = self.decode(dec_inputs, enc_inputs, enc_outputs)
        if visualize:
            return outputs, enc_attentions
        return outputs

    def encode(self, enc_inputs):
        enc_outputs, enc_attentions = self.encoder(enc_inputs)
        return enc_outputs, enc_attentions

    def decode(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        linear_outputs = self.linear(dec_outputs)
        outputs = nn.functional.log_softmax(linear_outputs, dim=-1)
        return outputs
