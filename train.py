from modules import AlzheimerModel
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import nibabel as nib
import nibabel.freesurfer.mghformat as mgh
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import torch
import pickle as pkl
import numpy as np
import pandas as pd
import argparse
import time
import os
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
dtype=torch.float32

idct_list = ['cdrsb','adasscore','mmscore','faqtotal','limmtotal','ldeltotal',
    'cerebrum_tcc','cerebrum_white','left_hippo','right_hippo','total_hippo']


def one_hot(str):
    if str == 'MRI':
        return np.array([1, 0, 0], dtype=np.float32)
    if str == 'CN':
        return np.array([0, 1, 0], dtype=np.float32)
    if str == 'AD':
        return np.array([0, 0, 1], dtype=np.float32)
    

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

        self.labels = []
        self.images = []
        self.indicators = []

        data = pd.read_csv('selected_data.csv').fillna(0)
        for i, ptid in enumerate(data['ptid']):
            path = os.popen(f"find /home/comp5331/ADNIp/{ptid} -name '*_wm.mgz'", 'r').read()[:-1]
            if path != '':
                self.labels.append(one_hot(data['label'][i]))
                self.indicators.append(data[i:i+1][idct_list].to_numpy(dtype=np.float32))
                image = np.asanyarray(nib.load(path).dataobj, dtype=np.float32)
                cut = int((image.shape[0]-128)/2)
                image = image[:image.shape[0]-cut, :192, :192][cut:, 64:, 64:]
                self.images.append(np.expand_dims(image, axis=0))

    def __len__(self):
        'Denotes the number of samples'
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        indicator = self.indicators[idx]
        label = self.labels[idx]

        return image, indicator, label


def main(args):
    out_dir = f'{args.epochs}_{args.batch_size}_{args.lr}_{args.l2_regular}'

    # Load data
    print('=====> Preparing data...')
    data = AlzheimerDataset('.')
    data_size = len(data)
    print('data_size: ', data_size)
    train_size = int(data_size*0.7)
    valid_size = int(data_size*0.1)
    test_size = data_size-train_size-valid_size
    train_data, valid_data, test_data = torch.utils.data.random_split(
        data, [train_size, valid_size, test_size])
    train_loader, valid_loader, test_loader = [
        torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=(dataset!=test_data))
        for dataset in [train_data, valid_data, test_data]
    ]

    # Get model structure
    model = AlzheimerModel(indicator_dim=11).to(device=device, dtype=dtype)
    
    # Evaluate only
    if args.evaluate != '':
        model.load_state_dict(torch.load(args.evaluate), strict=True)
        print(f"====> Loaded parameters '{args.evaluate}', testing...")
        model.eval()
        gt, pred = [], []
        with torch.no_grad():
            for images, indicators, labels in train_loader:
                print(images)
                images = images.to(device)
                indicators = indicators.to(device)
                labels = labels.to(device)
                outputs = model(images, indicators)
                gt.append(labels.cpu())
                pred.append(outputs.cpu())
            gt = torch.concat(gt, dim=0)
            pred = torch.concat(pred, dim=0)
            metrics = get_metrics(gt, pred)
            print(pred)
            print_dict(metrics)
        return

    # Optmizer and Loss function
    writer = SummaryWriter(log_dir=os.path.join(args.out, out_dir))
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_regular)

    # Training
    print('=====> Training...')
    print(f'{train_size} training data in total!')
    print(f'{valid_size} validation data in total!')
    val_min = 1000
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}', end='\t')
        start_time = time.time()

        # Training set
        model.train()
        for images, indicators, labels in train_loader:
            images = images.to(device)
            indicators = indicators.to(device)
            labels = labels.to(device)
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
                images = images.to(device)
                indicators = indicators.to(device)
                labels = labels.to(device)
                outputs = model(images, indicators)
                gt.append(labels.cpu())
                pred.append(outputs.cpu())
            gt = torch.concat(gt, dim=0)
            pred = torch.concat(pred, dim=0)
            val_loss = loss_fn(pred, gt).cpu().item()
        if val_loss < val_min:
            val_min = val_loss
            torch.save(model.state_dict(), os.path.join(args.out, f'checkpoint.pt'))

        print(f'loss: {loss:.5f}\t', f'val_loss: {val_loss:.5f}', end='\t')
            
        # Record time
        end_time =time.time()
        print(f'({end_time-start_time:.2f}s)')

        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)

    # Save model parameters
    torch.save(model.state_dict(), os.path.join(args.out, 'final.pt'))

    # Testing
    print('=====> Testing...')
    print(f'{test_size} testing data in total!')
    model.eval()
    gt, pred = [], []
    with torch.no_grad():
        for images, indicators, labels in test_loader:
            images = images.to(device)
            indicators = indicators.to(device)
            labels = labels.to(device)
            outputs = model(images, indicators)
            gt.append(labels.cpu())
            pred.append(outputs.cpu())
        gt = torch.concat(gt, dim=0)
        pred = torch.concat(pred, dim=0)
        metrics = get_metrics(gt, pred)
        print_dict(metrics)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default='logs', help='output directory')
    parser.add_argument('--evaluate', type=str, default='', help='evaluate only flag, must specify param file')
    parser.add_argument('--epochs', type=int, default=220)    
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--l2_regular', type=float, default=0.004)
    
    args=parser.parse_args()
    main(args)