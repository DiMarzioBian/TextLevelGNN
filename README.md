# TextLevelGNN

This is a toy PyTorch implementation for the EMNLP'19  paper [Text Level Graph Neural Network for Text Classification](https://www.aclweb.org/anthology/D19-1345.pdf)

---
## Training

Currently support `DATASET_NAME`: R8, R52, Ohsumed, MR.
Preprocessed corpus and GloVe are saved in `.pkl` files under `./data/`.

#### To be noticed, set `args.max_len_text` to 150 for Ohsumed, and 100 for all others.

---
## Dataset

#### If you wanna re-preprocess the data, please read below.

Please refer to [textGCN](https://github.com/yao8839836/text_gcn/tree/master/data) and copy the **R8, R52, mr, ohsumed_single_23** folder into data folder.

For word embeddings, please refer to [GloVe](https://nlp.stanford.edu/projects/glove/), 
download [glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip), 
unzip and move the file `glove.6B.300d.txt` to `./data/_pretrained/`

---
## Results

We are only able to implement similar result on R8 and R52 dataset, while Ohsumed perform quite difference as compared to paper's one.

| Accuracy | R8     | R52    | Ohsumed | MR     |
|----------|--------|--------|---------|--------|
| Train    | 99.45% | 96.27% | 95.99%  | 99.28% |
| Valid    | 97.27% | 93.73% | 52.68%  | 75.33% |
| Test     | 96.30% | 92.72% | 53.97%  | 73.21% |
