import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, args, data, gt):
        self.data = data
        self.gt = gt
        self.length = len(self.gt)

        self.n_word = args.n_word
        self.n_degree = args.n_degree
        self.max_len_text = args.max_len_text

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        text_tokens = self.data[idx]
        # text_tokens = self.data[idx][:self.max_len_text]
        len_text = len(text_tokens)
        nb_text = []

        for idx_token in range(len_text):
            nb_front, nb_tail = [], []
            for i in range(self.n_degree):
                before_idx = idx_token - 1 - i
                nb_front.append(text_tokens[before_idx] if before_idx > -1 else 0)
                after_idx = idx_token + 1 + i
                nb_tail.append(text_tokens[after_idx] if after_idx < len_text else 0)

            nb_text.append(nb_front + nb_tail)

        # pad to self.max_len_text
        x = np.zeros(self.max_len_text, dtype=np.int)
        x[:min(len(text_tokens), self.max_len_text)] = np.array(text_tokens)[: self.max_len_text]

        nb_x = np.zeros((self.max_len_text, self.n_degree * 2), dtype=np.int)
        nb_x[:min(len(nb_text), self.max_len_text)] = np.array(nb_text)[:self.max_len_text]

        w_edge_head_idx = ((x-1) * self.n_word).reshape(-1, 1)
        w_edge = w_edge_head_idx + nb_x
        w_edge[x == 0] = 0

        return x, nb_x, w_edge, self.gt[idx]


def collate_fn(insts):
    """ Batch preprocess """
    x, nb_x, w_edge, y = list(zip(*insts))

    x = torch.LongTensor(np.array(x))
    nb_x = torch.LongTensor(np.array(nb_x))
    w_edge = torch.LongTensor(np.array(w_edge))
    y = torch.LongTensor(y)
    return x, nb_x, w_edge, y


def get_dataloader(args):
    """ Get dataloader, word2idx and pretrained embeddings """

    with open(args.path_data, 'rb') as f:
        mappings = pickle.load(f)

    # label2idx = mappings['label2idx']
    word2idx = mappings['word2idx']
    tr_data = mappings['tr_data']
    tr_gt = mappings['tr_gt']
    val_data = mappings['val_data']
    val_gt = mappings['val_gt']
    te_data = mappings['te_data']
    te_gt = mappings['te_gt']
    embeds = mappings['embeds']
    args_prepare = mappings['args']

    if args_prepare.d_pretrained != args.d_model:
        raise ValueError('Experiment settings do not match data preprocess settings. '
                         'Please re-run prepare.py with correct settings.')
    args.n_class = args_prepare.n_class

    args.n_word = len(word2idx)  # including <pad> and <unk>

    train_loader = DataLoader(TextDataset(args, tr_data, tr_gt), batch_size=args.batch_size,
                              num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)

    valid_loader = DataLoader(TextDataset(args, val_data, val_gt), batch_size=args.batch_size,
                              num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)

    test_loader = DataLoader(TextDataset(args, te_data, te_gt), batch_size=args.batch_size,
                             num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)

    return train_loader, valid_loader, test_loader, word2idx, torch.Tensor(embeds)
