# custom_rnn.py
import torch
from torch import nn


class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, activation="tanh"):
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.activation = self._get_activation(activation)

        # Embedding layer: h_t -> m_t
        self.embedding = nn.Linear(hidden_size, embed_size)

        # Recurrent layer: input_t + m_t-1 -> h_t
        self.input_to_hidden = nn.Linear(input_size + embed_size, hidden_size)

    def _get_activation(self, activation):
        if activation == "tanh":
            return torch.tanh
        elif activation == "relu":
            return torch.relu
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, input_t, h_prev, m_prev):
        # Compute the embedding m_t = f(h_prev)
        m_t = self.embedding(h_prev)

        # Concatenate input_t and m_prev
        combined = torch.cat((input_t, m_prev), dim=1)

        # Compute the next hidden state h_t
        h_t = self.activation(self.input_to_hidden(combined))

        return h_t, m_t


class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, activation="tanh"):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.cell = CustomRNNCell(input_size, hidden_size, embed_size, activation)

    def forward(self, input_seq, h_0=None, m_0=None):
        seq_len, batch_size, _ = input_seq.size()

        if h_0 is None:
            h_t = input_seq.new_zeros(batch_size, self.hidden_size)
        else:
            h_t = h_0

        if m_0 is None:
            m_t = input_seq.new_zeros(batch_size, self.embed_size)
        else:
            m_t = m_0

        outputs = []
        for t in range(seq_len):
            input_t = input_seq[t]
            h_t, m_t = self.cell(input_t, h_t, m_t)
            outputs.append(h_t.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, h_t, m_t
