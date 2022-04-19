import argparse
import numpy as np
import codecs
import pickle
import time
import zipfile


def main():
    parser = argparse.ArgumentParser(description='TextLevelGNN-DGL data packaging project')

    # experiment setting
    parser.add_argument('--dataset', type=str, default='ohsumed', choices=['r8', 'r52', 'ohsumed'], help='used dataset')
    parser.add_argument('--pretrained', type=bool, default=True, help='use pretrained GloVe embeddings')
    parser.add_argument('--d_pretrained', type=int, default=300, help='pretrained embedding dimension')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    # path settings
    parser.add_argument('--path_data', type=str, default='./data/', help='path of the data corpus')

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset not in ['r8', 'r52', 'ohsumed']:
        raise ValueError('Data {data} not supported, currently supports "r8", "r52" and "ohsumed".')

    # read files
    print('\n[info] Dataset:', args.dataset)
    time_start = time.time()

    label2idx = read_label(args.path_data + args.dataset + '/label.txt')
    word2idx = read_vocab(args.path_data + args.dataset + '/vocab-5.txt')
    args.n_class = len(label2idx)
    args.n_word = len(word2idx)
    print('\tTotal classes:', args.n_class)
    print('\tTotal words:', args.n_word)

    embeds = get_embedding(args, word2idx)

    tr_data, tr_gt = read_corpus(args.path_data + args.dataset + '/train-stemmed.txt', label2idx, word2idx)
    print('\n\tTotal training samples:', len(tr_data))

    val_data, val_gt = read_corpus(args.path_data + args.dataset + '/valid-stemmed.txt', label2idx, word2idx)
    print('\tTotal validation samples:', len(val_data))

    te_data, te_gt = read_corpus(args.path_data + args.dataset + '/test-stemmed.txt', label2idx, word2idx)
    print('\tTotal testing samples:', len(te_data))

    # save processed data
    mappings = {
        'label2idx': label2idx,
        'word2idx': word2idx,
        'tr_data': tr_data,
        'tr_gt': tr_gt,
        'val_data': val_data,
        'val_gt': val_gt,
        'te_data': te_data,
        'te_gt': te_gt,
        'embeds': embeds,
        'args': args
    }

    with open(args.path_data + args.dataset + '.pkl', 'wb') as f:
        pickle.dump(mappings, f)
    with zipfile.ZipFile(args.path_data + args.dataset + '.zip', 'w') as zf:
        zf.write(args.path_data + args.dataset + '.pkl', compress_type=zipfile.ZIP_DEFLATED)

    print('\n[info] Time consumed: {:.2f}s'.format(time.time() - time_start))


def read_label(path):
    """ Extract and encode labels. """
    with open(path) as f:
        labels = f.read().split('\n')

    return {label: i for i, label in enumerate(labels)}


def read_vocab(path):
    """ Extract words from vocab and encode. """
    with open(path) as f:
        words = f.read().split('\n')
    word2idx = {word: i + 1 for i, word in enumerate(words)}
    word2idx['<pad>'] = 0

    return word2idx


def read_corpus(path, label2idx, word2idx):
    """ Encode both corpus and labels. """
    with open(path) as f:
        content = [line.split('\t') for line in f.read().split('\n')]

    data = [[encode_word(word, word2idx) for word in x[1].split()] for x in content]
    gt = [label2idx[x[0]] for x in content]
    return data, gt


def encode_word(word, word2idx):
    """ Encode word considering unknown word. """
    try:
        idx = word2idx[word]
    except KeyError:
        idx = word2idx['UNK']
    return idx


def get_embedding(args, word2idx):
    """ Find words in pretrained GloVe embeddings. """
    if args.pretrained:
        path = args.path_data + 'glove.6B.' + str(args.d_pretrained) + 'd.txt'
        embeds_word = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word2idx), args.d_pretrained))
        emb_counts = 0
        for i, line in enumerate(codecs.open(path, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == (args.d_pretrained + 1) and s[0] in word2idx:
                embeds_word[word2idx[s[0]]] = np.array([float(i) for i in s[1:]])
                emb_counts += 1

        embeds_word[0] = np.zeros_like(embeds_word[0])  # <pad>

    else:
        embeds_word = None
        emb_counts = 'disabled pretrained'

    print('\tPretrained GloVe found:', emb_counts)
    return embeds_word


if __name__ == '__main__':
    main()
