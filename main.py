import argparse
import os
import time
import copy
import numpy as np
import torch

from dataloader import get_dataloader
from TextLevelGNN.model import TextLevelGNN
from epoch import train, evaluate


def main():
    parser = argparse.ArgumentParser(description='TextLevelGNN project')

    # experiment setting
    parser.add_argument('--dataset', type=str, default='ohsumed', choices=['mr', 'ohsumed', 'r8', 'r52'],
                        help='dataset name')
    parser.add_argument('--mean_reduction', action='store_false', help='ablation: use mean reduction instead of max')
    parser.add_argument('--pretrained', action='store_true', help='ablation: use pretrained GloVe')
    parser.add_argument('--layer_norm', action='store_false', help='ablation: use layer normalization')
    parser.add_argument('--relu', action='store_true', help='ablation: use relu before softmax')
    parser.add_argument('--n_degree', type=int, default=99, help='neighbor region radius')

    # hyperparameters
    parser.add_argument('--d_model', type=int, default=300, help='node representation dimensions including embedding')
    parser.add_argument('--max_len_text', type=int, default=100,
                        help='maximum length of text, default 100, and 150 for ohsumed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for computing')

    # training settings
    parser.add_argument('--num_worker', type=int, default=5, help='number of dataloader worker')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate applied to layers (0 = no dropout)')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--lr_step', type=int, default=5, help='number of epoch for each lr downgrade')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='strength of lr downgrade')
    parser.add_argument('--es_patience_max', type=int, default=5, help='max early stopped patience')
    parser.add_argument('--loss_eps', type=float, default=1e-4, help='minimum loss change threshold')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    # path settings
    parser.add_argument('--path_data', type=str, default='./data/', help='path of the data corpus')
    parser.add_argument('--path_log', type=str, default='./result/logs/', help='path of the training logs')
    parser.add_argument('--path_model', type=str, default='./result/models/', help='path of the trained model')
    parser.add_argument('--save_model', type=bool, default=False, help='save model for further use')

    args = parser.parse_args()
    args.device = torch.device(args.device)

    if not os.path.exists(args.path_data + args.dataset + '.pkl'):
        raise FileNotFoundError('Processed "{data}" pkl file not found, '
                                'please download corpus & GloVe and run prepare.py first.'
                                .format(data=args.dataset))

    for path in [args.path_log, args.path_model]:
        if not os.path.exists(path):
            os.makedirs(path)

    args.path_model_params = args.path_model + 'model_params' + time.strftime('_%b_%d_%H_%M', time.localtime()) + '.pt'
    args.path_model += 'model_cuda' + str(args.device)[-1] + time.strftime('_%b_%d_%H_%M', time.localtime()) + '.pt'
    args.path_log += 'log' + time.strftime('_%b_%d_%H_%M', time.localtime()) + '.txt'
    args.path_data += args.dataset + '.pkl'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # prepare data and model
    print('\n[info] Project starts.')
    train_loader, valid_loader, test_loader, word2idx, embeds_pretrained = get_dataloader(args)

    model = TextLevelGNN(args, embeds_pretrained).to(args.device)

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # Start modeling
    print('\n[info] | Dataset: {Dataset} | mean_reduction: {mean_reduction} | pretrained: {pretrained} '
          '| n_degree: {n_degree} | layer_norm: {layer_norm} | relu: {relu} |'
          .format(Dataset=args.dataset, mean_reduction=args.mean_reduction, pretrained=args.pretrained,
                  n_degree=args.n_degree, layer_norm=args.layer_norm, relu=args.relu))
    loss_best = 1e5
    acc_best = 0
    epoch_best = 0
    es_patience = 0

    for epoch in range(1, args.epochs+1):
        print('\n[Epoch {epoch}]'.format(epoch=epoch))

        # training phase
        t_start = time.time()
        loss_train, acc_train = train(args, model, train_loader, optimizer)
        scheduler.step()
        print(' \t| Train | loss {:5.4f} | acc {:5.4f} | {:5.2f} s |'
              .format(loss_train, acc_train, time.time() - t_start))

        # validating phase
        loss_val, acc_val = evaluate(args, model, valid_loader)

        # early stopping condition
        if acc_val > acc_best or (acc_val == acc_best and loss_best - loss_val > args.loss_eps):
            es_patience = 0
            state_best = copy.deepcopy(model.state_dict())
            loss_best = loss_val
            acc_best = acc_val
            epoch_best = epoch
        else:
            es_patience += 1
            if es_patience >= args.es_patience_max:
                print('\n[Warning] Early stopping model')
                print('\t| Best | epoch {:d} | loss {:5.4f} | acc {:5.4f} |'
                      .format(epoch_best, loss_best, acc_best))
                break

        # logging
        print('\t| Valid | loss {:5.4f} | acc {:5.4f} | es_patience {:.0f}/{:.0f} |'
              .format(loss_val, acc_val, es_patience, args.es_patience_max))

    # testing phase
    print('\n[Testing]')
    model.load_state_dict(state_best)
    if args.save_model:
        with open(args.path_model_params, 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(args.path_model, 'wb') as f:
            torch.save(model, f)

    loss_test, acc_test = evaluate(args, model, test_loader)

    print('\n\t| Test | loss {:5.4f} | acc {:5.4f} |'
          .format(loss_test, acc_test))
    print('\n[info] | Dataset: {Dataset} | mean_reduction: {mean_reduction} | pretrained: {pretrained} '
          '| n_degree: {n_degree} | layer_norm: {layer_norm} | relu: {relu} |\n'
          .format(Dataset=args.dataset, mean_reduction=args.mean_reduction, pretrained=args.pretrained,
                  n_degree=args.n_degree, layer_norm=args.layer_norm, relu=args.relu))


if __name__ == '__main__':
    main()
