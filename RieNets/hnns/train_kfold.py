from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import torch

import geoopt
import RieNets.hnns.optimizers as optimizers
from torch.utils.tensorboard import SummaryWriter

from .models.base_models import NCModel, LPModel
from .utils.data_utils import load_data
from .utils.train_utils import format_metrics,set_seed,parse_cfg,write_final_results

def train_kfold(cfg,args):
    args = parse_cfg(args, cfg)
    set_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    # set logger
    logger = logging.getLogger(args.modelname)
    logger.setLevel(logging.INFO)

    args.logger = logger
    logger.info('Dataset: {}'.format(args.modelname, args.dataset))
    logger.info(f'Using: {args.device}')
    logger.info("Using seed {}.".format(args.seed))

    fold_metrics=[]
    for ith_fold in range(args.folds):
        ith_best_test_metrics = train_process(args,ith_fold,logger)
        fold_metrics.append(ith_best_test_metrics)

    # calculate the average metrics
    loss_values = [metrics['loss'].item() for metrics in fold_metrics]
    roc_values = [metrics['roc'] for metrics in fold_metrics]
    ap_values = [metrics['ap'] for metrics in fold_metrics]

    # Compute the mean and standard deviation
    average_loss = np.mean(loss_values)
    std_loss = np.std(loss_values)
    average_roc = np.mean(roc_values)
    std_roc = np.std(roc_values)
    average_ap = np.mean(ap_values)
    std_ap = np.std(ap_values)

    message = f"{args.folds}-folds: " \
              f"average best loss: {average_loss * 100:.2f} ± {std_loss * 100:.2f}, " \
              f"average best ROC: {average_roc*100:.2f} ± {std_roc*100:.2f}, " \
              f"average best AP: {average_ap*100:.2f} ± {std_ap*100:.2f}"
    logger.info(message)
    write_final_results(args.dataset,args.modelname+' '+message)


def train_process(args,ith_fold,logger):
    # setting writer
    if args.is_writer:
        # if args.folds>1:
        #     args.writer_path = os.path.join('./tensorboard_logs', f"{args.modelname}_{args.ith_fold}")
        # else:
        #     args.writer_path = os.path.join('./tensorboard_logs/', f"{args.modelname}_{args.ith_fold}")
        args.writer_path = os.path.join('./tensorboard_logs/', f"{args.modelname}_{ith_fold+1}")
        logger.info('writer path {}'.format(args.writer_path))
        args.writer = SummaryWriter(args.writer_path)

    # Load data and model
    data = load_data(args, os.path.join(args.path, args.dataset))
    # data = load_data(args, os.path.join('/nfs/data_todi/zchen/Proposed_Methods/GyroBNH/data', args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logger.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    # optimizer = geoopt.optim.RiemannianAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    logger.info(str(model))
    logger.info(optimizer)
    logger.info(lr_scheduler)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    training_time = []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        training_time.append(time.time() - t)
        if (epoch + 1) % args.log_freq == 0:
            logger.info(" ".join([f'Train: Fold: {ith_fold+1}/{args.folds}, Epoch:{epoch + 1}/{args.epochs}',
                                  'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                  format_metrics(train_metrics),
                                  'time: {:.4f}s'.format(training_time[-1])
                                  ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            if (epoch + 1) % args.log_freq == 0:
                logger.info(" ".join([f'Val: Fold: {ith_fold+1}/{args.folds}, Epoch:{epoch + 1}/{args.epochs}', format_metrics(val_metrics)]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logger.info("Early stopping")
                    break

        # save data into tensorboard
        if args.is_writer:
            args.writer.add_scalar('loss/val', val_metrics['loss'].item(), epoch)
            args.writer.add_scalar('roc/val', val_metrics['roc']*100, epoch)
            args.writer.add_scalar('ap/val', val_metrics['ap']*100, epoch)
            args.writer.add_scalar('loss/train', train_metrics['loss'].item(), epoch)
            args.writer.add_scalar('roc/train', train_metrics['roc']*100, epoch)
            args.writer.add_scalar('ap/train', train_metrics['ap']*100, epoch)
            args.writer.add_scalar('loss/best_test', best_test_metrics['loss'].item(), epoch)
            args.writer.add_scalar('roc/best_test', best_test_metrics['roc']*100, epoch)
            args.writer.add_scalar('ap/best_test', best_test_metrics['ap']*100, epoch)

    logger.info("Optimization Finished!")
    lastk_average_time = np.asarray(training_time[-5:]).mean()
    mink_average_time = np.asarray(sorted(training_time)[:5]).mean()
    logger.info(f"Total time elapsed: {time.time() - t_total:.4f}s "
                f"with average time: {lastk_average_time:.4f} "
                f"and average smallest time: {mink_average_time:.4f}")
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    logger.info(" ".join(["Val set results:", format_metrics(best_val_metrics)]))
    logger.info(" ".join(["Test set results:", format_metrics(best_test_metrics)]))
    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logger.info(f"Saved model in {save_dir}")

    return best_test_metrics