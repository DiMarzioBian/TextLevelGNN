import torch
from torch import nn
import torch.nn.functional as F


class TextLevelGNN(nn.Module):

    def __init__(self, args, embeds_pretrained):
        super(TextLevelGNN, self).__init__()
        self.n_node = args.n_word
        self.d_model = args.d_model
        self.n_category = args.n_category

        if args.pretrained:
            self.embedding = nn.Embedding.from_pretrained(embeds_pretrained, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(self.n_node, self.d_model, padding_idx=0)
            nn.init.xavier_uniform_(self.embedding.weight)

        self.edge_weights = nn.Embedding((self.n_node - 1) * self.n_node + 1, 1, padding_idx=0)  # +1 for padding
        self.node_weights = nn.Embedding(self.n_node, 1, padding_idx=0)  # Nn, node weight for itself
        nn.init.xavier_uniform_(self.edge_weights.weight)
        nn.init.xavier_uniform_(self.node_weights.weight)

        self.fc = nn.Linear(self.d_model, self.n_category, bias=True)

    def forward(self, x, nb_x, w_edge):
        """
        INPUT:
            x: nodes of a sentence, (batch, sentence_maxlen)
            nb_x: neighbor nodes of node x, (batch, max_len_text, n_gram*2):
            w_edge: neighbor weights of neighbor nodes of node x, (batch, max_len_text, n_gram*2)

        OUTPUT:
            y: Predicted Probabilities of each classes
        """
        # Neighbor
        # Neighbor Messages (Mn)
        msg_nb = self.embedding(nb_x)  # (BATCH, SEQ_LEN, NEIGHBOR_SIZE, EMBED_DIM)

        # EDGE WEIGHTS
        w_edge = self.edge_weights(w_edge)  # (BATCH, SEQ_LEN, NEIGHBOR_SIZE )

        # get representation of Neighbors
        msg_nb = torch.max(w_edge * msg_nb, dim=2).values  # (BATCH, SEQ_LEN, EMBED_DIM)

        # Self Features (Rn)
        emb_node = self.embedding(x)  # (BATCH, SEQ_LEN, EMBED_DIM)

        # Aggregate information from neighbor
        # get self node weight (Nn)
        w_node = self.node_weights(x)

        Rn = (1 - w_node) * msg_nb + w_node * emb_node

        # Aggregate node features for sentence
        X = torch.sum(Rn, dim=1)

        y = self.fc(X)
        return y
