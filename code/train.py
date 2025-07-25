import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import logging

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

from our_model import create_model
from dataset import Oredataset
from torch.utils.data import DataLoader
from utils.WarmUpLR import WarmupLR


# Set the random number seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

GLOBAL_SEED = 0
GLOBAL_WORKER_ID = None

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def get_args():
    parser = argparse.ArgumentParser(description='Ore classification')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--init_lr', type=float, default=1e-5, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--num_warmup', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--warmup_strategy', type=str, default='cos', help='Warmup strategy')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument("--in_dim",type=int, default=1, help='Input dimension') 
    parser.add_argument("--seed", type=int, default=1258, help='Random seed')
    parser.add_argument("--pretrained", type=str, default= None , help='Pretrained model weight path')
    parser.add_argument('--freeze-layers', type=bool, default=False, help='Freezing sorting head')
    parser.add_argument('--include_top', type=bool, default=False, help='Whether average pooling and classification headers are included')
    parser.add_argument('--gpu_id', type=int, nargs='+', default=[0,1], help='GPU IDs to use')
    parser.add_argument("--savePath", type=str, default='Your path')
    parser.add_argument("--json_path",type=str, default='Your path/train_val_test.json')
    parser.add_argument("--modelName",type=str,default='dual_tower', choices=['dual_tower', 'dual_tower_high', 'dual_tower_low'])
    return parser.parse_args()

def train(model, dataloader, dataloader_test, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # If there are multiple Gpus, use DataParallel
    if torch.cuda.device_count() > 1 and len(args.gpu_id) > 1:
        print(f"使用 {len(args.gpu_id)} 个GPU:{args.gpu_id}")
        model = nn.DataParallel(model, device_ids=args.gpu_id)

    # Set logs
    if not os.path.exists(args.savePath):
        os.makedirs(args.savePath)
    log_file = os.path.join(args.savePath, 'training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info(args)
    logging.info(f"Using {torch.cuda.device_count()} GPUs: {args.gpu_id}")
    logging.info("Model Summary:")
    logging.info(model)
    logging.info('-' * 100)

    # Set the learning rate strategy and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
    scheduler_steplr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.min_lr) #设置 eta_min 的目的是为了保证在训练的后期，即使余弦退火函数的值变得非常小，学习率也不会无限接近于零。这样可以避免模型在训练结束时陷入局部最优或者难以继续学习的状态。
    scheduler = WarmupLR(scheduler_steplr, init_lr=args.init_lr, num_warmup=args.num_warmup, warmup_strategy=args.warmup_strategy)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    best_f1 = -1
    for epoch in range(args.num_epochs):
        print('-' * 100)
        print(f'Epoch {epoch + 1}/{args.num_epochs}')
        
        # Training cycle
        model.train()
        train_loss = 0.0
        tra_all_labels = []
        tra_all_preds = []
        tra_all_probs = []
        scheduler.step()
        for data in dataloader:
            input_low, input_high, labels = data
            input_low, input_high, labels = input_low.to(device), input_high.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_low, input_high)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            tra_all_labels.extend(labels.cpu().numpy())
            tra_all_preds.extend(preds.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)[:, 1].detach()
            tra_all_probs.extend(probs.cpu().numpy())

        # Verification cycle
        model.eval()
        val_loss = 0.0
        val_all_labels = []
        val_all_preds = []
        val_all_probs = []
        with torch.no_grad():
            for data in dataloader_test:
                input_low, input_high, labels = data
                input_low, input_high, labels = input_low.to(device), input_high.to(device), labels.to(device)
                outputs = model(input_low, input_high)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_all_labels.extend(labels.cpu().numpy())
                val_all_preds.extend(preds.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)[:, 1].detach()  #The positive category is 1
                val_all_probs.extend(probs.cpu().numpy())

                
        tra_accuracy = accuracy_score(tra_all_labels, tra_all_preds)
        tra_precision = precision_score(tra_all_labels, tra_all_preds, average='binary')
        tra_recall = recall_score(tra_all_labels, tra_all_preds, average='binary')
        tra_f1 = f1_score(tra_all_labels, tra_all_preds, average='binary')
        try:
            tra_auc = roc_auc_score(tra_all_labels, tra_all_probs)
            tra_PR_auc = average_precision_score(tra_all_labels, tra_all_probs)
            
            tra_P, tra_R, threshold = precision_recall_curve(tra_all_labels, tra_all_probs)
            tra_pr_auc0 = auc(tra_R, tra_P)
        except:
            tra_auc = 0.0
            tra_PR_auc = 0.0
            tra_pr_auc0 = 0.0

        tra_confusion_matrix1 = confusion_matrix(tra_all_labels, tra_all_preds)

        val_accuracy = accuracy_score(val_all_labels, val_all_preds)
        val_precision = precision_score(val_all_labels, val_all_preds, average='binary')
        val_recall = recall_score(val_all_labels, val_all_preds, average='binary')
        val_f1 = f1_score(val_all_labels, val_all_preds, average='binary')
        try:
            val_auc = roc_auc_score(val_all_labels, val_all_probs)
            val_PR_auc = average_precision_score(val_all_labels, val_all_probs)

            val_P, val_R, threshold = precision_recall_curve(val_all_labels, val_all_probs)
            val_pr_auc0 = auc(val_R, val_P)
        except:
            val_auc = 0.0
            val_PR_auc = 0.0
            val_pr_auc0 = 0.0

        val_confusion_matrix1 = confusion_matrix(val_all_labels, val_all_preds)

        # Print and record logs
        print(f'Epoch {epoch}:\n -----Train Loss: {train_loss/len(dataloader):.6f}\n Accuracy: {tra_accuracy:.6f},  Precision: {tra_precision:.6f},  Recall: {tra_recall:.6f},  F1: {tra_f1:.6f},  AUC: {tra_auc:.6f},  PR_AUC: {tra_PR_auc:.6f},  PR_AUC0: {tra_pr_auc0:.6f}')
        print(f'matrix:{np.array2string(tra_confusion_matrix1)}\n')
        print(f'-----Val Loss: {val_loss/len(dataloader_test):.6f}\n Accuracy: {val_accuracy:.6f},  Precision: {val_precision:.6f},  Recall: {val_recall:.6f},  F1: {val_f1:.6f},  AUC: {val_auc:.6f},  PR_AUC: {val_PR_auc:.6f},  PR_AUC0: {val_pr_auc0:.6f}')
        print(f'matrix:{np.array2string(val_confusion_matrix1)}\n')
        with open(log_file, 'a') as f:
            f.write(f'Epoch {epoch}:\n -----Train Loss: {train_loss/len(dataloader):.6f}\n Accuracy: {tra_accuracy:.6f},  Precision: {tra_precision:.6f},  Recall: {tra_recall:.6f},  F1: {tra_f1:.6f},  AUC: {tra_auc:.6f},  PR_AUC: {tra_PR_auc:.6f},  PR_AUC0: {tra_pr_auc0:.6f}\n')
            f.write(f'matrix:{np.array2string(tra_confusion_matrix1)}\n')
            f.write(f'-----Val Loss: {val_loss/len(dataloader_test):.6f}\n Accuracy: {val_accuracy:.6f},  Precision: {val_precision:.6f},  Recall: {val_recall:.6f},  F1: {val_f1:.6f},  AUC: {val_auc:.6f},  PR_AUC: {val_PR_auc:.6f},  PR_AUC0: {val_pr_auc0:.6f}\n')
            f.write(f'matrix:{np.array2string(val_confusion_matrix1)}\n')

        train_losses.append(train_loss / len(dataloader))
        val_losses.append(val_loss / len(dataloader_test))

        # Plot the loss graph every twenty epochs
        if (epoch + 1) % 10 == 0:
            plt.figure()
            plt.plot(range(epoch+1), train_losses, label='Train Loss')
            plt.plot(range(epoch+1), val_losses, label='Val Loss')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.savefig(os.path.join(args.savePath, f'loss_epoch_{epoch+1}.png'))
            plt.close()

        # 保存模型
        save_dir = args.savePath
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'model_{}.pt'.format(epoch + 1))
        if val_f1 >= best_f1:
            best_f1 = val_f1
            state_dict = model.state_dict()
            os.system('rm ' + save_dir + '/*.pt')
            torch.save(state_dict, save_path)


def main(args):
    # 读取数据集
    datasetTrain = Oredataset(json_path = args.json_path, TrainValTest = 'train', transform='train')
    data_loaderTrain = DataLoader(datasetTrain, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    
    datasetVal = Oredataset(json_path=args.json_path, TrainValTest = 'val', transform=None)
    data_loaderVal = DataLoader(datasetVal, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

    model = create_model(args)

    # print(model)
    trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total trainable pytorch params:{} MB\n'.format(trainable_pytorch_total_params*1e-6))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total pytorch params:{} MB\n'.format(pytorch_total_params*1e-6))

    train(model, data_loaderTrain, data_loaderVal, args)



if __name__ == '__main__':
    args = get_args()

    worker_init_fn(args.seed)

    main(args)

