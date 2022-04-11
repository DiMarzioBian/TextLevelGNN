import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    def __init__(self, args, data, gt):
        self.data = data
        self.gt = gt
        self.length = len(self.gt)

        self.n_word = args.n_word
        self.n_gram = args.n_gram
        self.max_len_text = args.max_len_text

        args.n_category = len(set(self.gt))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        text_tokens = self.data[idx]
        len_text = len(text_tokens)
        nb_text = []

        for idx_token in range(len_text):
            nb_front, nb_tail = [], []
            for i in range(self.n_gram):
                before_idx = idx_token - 1 - i
                nb_front.append(text_tokens[before_idx] if before_idx > -1 else 0)
                after_idx = idx_token + 1 + i
                nb_tail.append(text_tokens[after_idx] if after_idx < len_text else 0)

            nb_text.append(nb_front + nb_tail)

        # pad to self.max_len_text
        x = np.zeros(self.max_len_text, dtype=np.int)
        x[:min(len(text_tokens), self.max_len_text)] = np.array(text_tokens)[: self.max_len_text]

        nb_x = np.zeros((self.max_len_text, self.n_gram * 2), dtype=np.int)
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


def load_pretrain_embedding(args):
    """ Load _pretrained embedding """
    with open(args.word_embedding_file, encoding="utf8") as f:
        lines = f.readlines()
        embedding = np.random.random((args.n_node, args.d_feature))
        for line in f.readlines():
            line_split = line.strip().split()
            if line_split[0] in args.word2idx:
                embedding[args.word2idx[line_split[0]]] = line_split[1:]

        embedding[0] = 0 # set the _PAD_ as 0
    return torch.tensor(embedding, dtype=torch.float)


def get_dataloader(args):
    """ Get dataloader, word2idx and pretrained embeddings """

    with open(args.path_data, 'rb') as f:
        mappings = pickle.load(f)

    train_data = mappings['tr_data']
    train_gt = mappings['tr_gt']
    test_x = mappings['te_data']
    test_y = mappings['te_gt']
    word2idx = mappings['word2idx']
    embeds = mappings['embeds']
    args_prepare = mappings['args']

    if args_prepare.max_len_text != args.max_len_text or args_prepare.d_pretrained != args.d_model:
        raise ValueError('Experiment settings do not match data preprocess settings. '
                         'Please re-run prepare.py with correct settings.')
    args.n_word = len(word2idx)  # including <pad> and <unk>

    train_x, valid_x, train_y, valid_y = train_test_split(train_data, train_gt, test_size=args.ratio_valid,
                                                          random_state=args.seed)

    train_loader = DataLoader(TextDataset(args, train_x, train_y), batch_size=args.batch_size,
                              num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)

    valid_loader = DataLoader(TextDataset(args, valid_x, valid_y), batch_size=args.batch_size,
                              num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)

    test_loader = DataLoader(TextDataset(args, test_x, test_y), batch_size=args.batch_size,
                             num_workers=args.num_worker, collate_fn=collate_fn, shuffle=True)

    return train_loader, valid_loader, test_loader, word2idx, torch.Tensor(embeds)
