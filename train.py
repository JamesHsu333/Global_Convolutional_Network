import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/JPEGImages',
                    help="Directory containing the dataset")
parser.add_argument('--mask_dir', default='data/SegmentationClass',
                    help="Directory containing the mask dataset")
parser.add_argument('--dataset_dir', default='data/',
                    help="Directory containing the train/val/test file names")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--num_classes', default=21,
                    help="Numbers of classes")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")

def train(model, optimizer, loss_fns, dataloader, metrics, params):
    # Set model to training mode
    model.train()

    # Summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
            labels_batch = labels_batch.float()

            # Forward
            output_batch = model(train_batch).float()

            # Backward
            loss = 0.4*loss_fns['BinaryCrossEntropy'](output_batch, labels_batch)+0.4*loss_fns['SoftDiceLoss'](output_batch, labels_batch)+0.2*loss_fns['SoftInvDiceLoss'](output_batch, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                summary_batch = {metric: metrics[metric](output_batch, labels_batch, 20+1)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fns, metrics, params, model_dir,
                       restore_file=None):
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fns, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fns, val_dataloader, metrics, params)

        val_acc = val_metrics['mIOU']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    
    params.cuda = torch.cuda.is_available()

    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
    
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info("Loading the datasets...")

    dataloader = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, args.mask_dir, args.dataset_dir, args.num_classes, params)

    train_dl = dataloader['train']
    val_dl = dataloader['val']

    logging.info("-done")

    model = net.FCN_GCN(args.num_classes).cuda() if params.cuda else net.FCN_GCN(args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.99, weight_decay=0.0005)

    loss_fns = net.loss_fns

    metrics = net.metrics

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fns, metrics, params, args.model_dir, args.restore_file)
