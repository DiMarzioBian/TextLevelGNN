from tqdm import tqdm
import torch
import torch.nn.functional as F


def train(args, model, data, optimizer):
    model.train()
    loss_total = 0.
    n_sample = 0
    correct_pred_total = 0

    for batch in tqdm(data, desc='  - training', leave=False):
        X, NX, EW, Y = map(lambda x: x.to(args.device), batch)

        optimizer.zero_grad()
        scores_batch = model(X, NX, EW)
        loss_batch = F.cross_entropy(scores_batch, Y)
        loss_batch.backward()
        optimizer.step()

        # calculate loss
        loss_total += loss_batch * scores_batch.shape[0]
        n_sample += scores_batch.shape[0]
        correct_pred_total += (scores_batch.max(dim=-1)[1] == Y).sum()

    loss_mean = loss_total / n_sample
    acc = correct_pred_total / n_sample

    return loss_mean, acc


def evaluate(args, model, data):
    model.train()
    loss_total = 0.
    n_sample = 0
    correct_pred_total = 0

    with torch.no_grad():
        for batch in tqdm(data, desc='  - evaluating', leave=False):
            X, NX, EW, Y = map(lambda x: x.to(args.device), batch)

            scores_batch = model(X, NX, EW)
            loss_batch = F.cross_entropy(scores_batch, Y)

            # calculate loss
            loss_total += loss_batch * scores_batch.shape[0]
            n_sample += scores_batch.shape[0]
            correct_pred_total += (scores_batch.max(dim=-1)[1] == Y).sum()

    loss_mean = loss_total / n_sample
    acc = correct_pred_total / n_sample

    return loss_mean, acc
