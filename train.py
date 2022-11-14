from typing import Mapping, List, Tuple, Sequence
from torch.utils.tensorboard import SummaryWriter
from modules import AlzheimerModel
from sklearn.metrics import confusion_matrix
import torch.utils.data
import torch.nn as nn
import torch
import pickle as pkl
import numpy as np
import pandas as pd
import argparse
import time
import os
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
dtype=torch.float32


def get_metrics(gt: torch.Tensor, pred: torch.Tensor):
    '''
        gt: (*, no_class)
        pred: (*, no_class)
    '''
    no_class = gt.size(1)
    gt = torch.max(gt, dim=1)[1]
    pred = torch.max(pred, dim=1)[1]
    con_mat = confusion_matrix(gt,pred)

    rec_sum, pre_sum, acc_sum, spe_sum = 0, 0, 0, 0
    for i in range(no_class):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        rec_sum += tp/(tp+fn)
        pre_sum += tp/(tp+fp)
        acc_sum += (tp+tn)/number
        spe_sum += tn/(tn+fp)
        
    return {
        'specificity': spe_sum/no_class,
        'recall': rec_sum/no_class,
        'accuracy': acc_sum/no_class,
        'precision': pre_sum/no_class,
        'F1': 2*pre_sum*rec_sum/(pre_sum+rec_sum)/no_class,
    }


def print_dict(dict_):
    for k, v in dict_.items():
        print(k, f':\t{v:.2f}')


class AlzheimerDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir):
        'Initialization'
        super(AlzheimerDataset, self).__init__()
        # Get labels
        # Get images
        # Get indicators

    def __len__(self):
        'Denotes the number of samples'
        return len(self.pids)

    def __getitem__(self, idx):
        image = self.images[idx]
        indicator = self.indicators[idx]
        label = self.labels[idx]

        return image, indicator, label


def main(args):
    # Hyperparameters
    epochs = 220
    batch_size = 6
    lr = 1e-6
    dropout = 0.02
    l2_regular = 0.004

    '''# Load data
    print('=====> Preparing data...')
    data = AlzheimerDataset('.')
    data_size = len(data)
    train_size = int(data_size*0.7)
    valid_size = int(data_size*0.1)
    test_size = data_size-train_size-valid_size
    train_data, valid_data, test_data = torch.utils.data.random_split(
        data, [train_size, valid_size, test_size])
    train_loader, valid_loader, test_loader = [
        torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(dataset!=test_data))
        for dataset in [train_data, valid_data, test_data]
    ]'''

    # Get model structure
    model = AlzheimerModel(indicator_dim=50)
    images = torch.zeros((5, 1, 128, 128, 128))
    indicators = torch.zeros((5, 1, 50))
    writer = SummaryWriter(log_dir=os.path.join(args.out, 'logs'))
    writer.add_graph(model, [images, indicators])
    model = model.to(device=device, dtype=dtype)
    
    # Evaluate only
    if args.evaluate:
        model.load_state_dict(torch.load(args.load_params), strict=True)
        print(f"====> Loaded parameters '{args.load_params}', testing...")
        model.eval()
        gt, pred = [], []
        with torch.no_grad():
            for images, indicators, labels in train_loader:
                outputs = model(images, indicators)
                gt.append(labels.cpu())
                pred.append(outputs.cpu())
            gt = torch.concat(gt, dim=0)
            pred = torch.concat(pred, dim=0)
            metrics = get_metrics(gt, pred)
            print_dict(metrics)

    # Optmizer and Loss function
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    print('=====> Training...')
    print(f'{train_size} training data in total!')
    print(f'{valid_size} validation data in total!')
    for epoch in range(args.epoch):
        print(f'Epoch {epoch+1}/{args.epoch}')
        start_time = time.time()

        # Training set
        model.train()
        for images, indicators, labels in train_loader:
            outputs = model(images, indicators)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = loss.cpu().item()
        
        # Validation set
        model.eval()
        gt, pred = [], []
        with torch.no_grad():
            for images, indicators, labels in valid_loader:
                outputs = model(images, indicators)
                gt.append(labels.cpu())
                pred.append(outputs.cpu())
            gt = torch.concat(gt, dim=0)
            pred = torch.concat(pred, dim=0)
            val_loss = loss_fn(pred, gt)
            
        # Record time
        end_time =time.time()
        print(f'({end_time-start_time:.2f}s)')

        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)

    # Testing
    print('=====> Testing...')
    print(f'{test_size} testing data in total!')
    model.eval()
    gt, pred = [], []
    with torch.no_grad():
        for images, indicators, labels in test_loader:
            outputs = model(images, indicators)
            gt.append(labels.cpu())
            pred.append(outputs.cpu())
        gt = torch.concat(gt, dim=0)
        pred = torch.concat(pred, dim=0)
        metrics = get_metrics(gt, pred)
        print_dict(metrics)

    # Save model parameters
    torch.save(model.state_dict(), os.path.join(args.out, f'final.pt'))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default='.', help='output directory')
    parser.add_argument('--evaluate', action='store_true', help='evaluate only flag, must specify param file')

    args=parser.parse_args()
    main(args)