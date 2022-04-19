import torch
from torch import nn
import torch.nn.functional as F


class TextLevelGNN(nn.Module):

    def __init__(self, args, embeds_pretrained):
        super(TextLevelGNN, self).__init__()
        self.n_node = args.n_word
        self.d_model = args.d_model
        self.n_class = args.n_class
        self.max_len_text = args.max_len_text
        self.layer_norm = args.layer_norm
        self.relu = args.relu
        self.mean_reduction = args.mean_reduction

        if args.pretrained:
            self.embedding = nn.Embedding.from_pretrained(embeds_pretrained, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(self.n_node, self.d_model, padding_idx=0)
            nn.init.xavier_uniform_(self.embedding.weight)

        if self.layer_norm:
            self.ln = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(args.dropout)

        self.weight_edge = nn.Embedding((self.n_node - 1) * self.n_node + 1, 1, padding_idx=0)  # +1 for padding
        self.eta_node = nn.Embedding(self.n_node, 1, padding_idx=0)  # Nn, node weight for itself
        nn.init.xavier_uniform_(self.weight_edge.weight)
        nn.init.xavier_uniform_(self.eta_node.weight)

        self.fc = nn.Linear(self.d_model, self.n_class, bias=True)

    def forward(self, x, nb_x, w_edge):
        """
        INPUT:
            x: nodes of a sentence, (batch, sentence_maxlen)
            nb_x: neighbor nodes of node x, (batch, max_len_text, n_degree*2):
            w_edge: neighbor weights of neighbor nodes of node x, (batch, max_len_text, n_degree*2)

        OUTPUT:
            y: Predicted Probabilities of each classes
        """

        idx_nodes = torch.cat((nb_x, x.unsqueeze(-1)), dim=-1)
        emb_nodes = self.embedding(idx_nodes)
        if self.layer_norm:
            emb_nodes = self.ln(emb_nodes.view(len(x), -1, self.d_model)).view(len(x), self.max_len_text, -1, self.d_model)

        # message generating
        w_edge = self.weight_edge(w_edge)
        if not self.mean_reduction:
            msg_nb = torch.max(w_edge * emb_nodes[:, :, :-1, :], dim=2).values
        else:
            msg_nb = torch.mean(w_edge * emb_nodes[:, :, :-1, :], dim=2)


        # message passing
        eta = self.eta_node(x)
        h_node = (1 - eta) * msg_nb + eta * emb_nodes[:, :, -1, :]

        scores = self.dropout(h_node.sum(dim=1))
        if self.relu:
            scores = F.relu(scores)
        scores = self.fc(scores)
        return scores
