from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from eval_metrics import eval_sysu, eval_regdb, evaluate
from model_main import embed_net


from utils import *
from loss import OriTripletLoss, CenterTripletLoss,CrossEntropyLabelSmooth, DC
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import math
from re_rank import random_walk, k_reciprocal
from data_manager import VCM
from data_loader import VideoDataset_train, VideoDataset_test, VideoDataset_train_evaluation
import transforms as T
import time
from datetime import datetime
# import cv2
import numpy as np
import openTSNE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing, ChannelExchange

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='VCM', help='dataset name: VCM(Video Cross-modal)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet50')
parser.add_argument('--resume', '-r', default='VCMCST_CTTM_MTFM_DC_2_t2v_rank1_best.t', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='/', type=str,
                    help='model save path')

parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=2048, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=4, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--part', default=3, type=int,
                    metavar='tb', help=' part number')
parser.add_argument('--method', default='id+tri', type=str,
                    metavar='m', help='method type')
parser.add_argument('--drop', default=0.2, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=2, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='2', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--lambda0', default=1.0, type=float,
                    metavar='lambda0', help='graph attention weights')
parser.add_argument('--graph', action='store_true', help='either add graph attention or not')
parser.add_argument('--wpa', action='store_true', help='either add weighted part attention')
parser.add_argument('--a', default=1, type=float,
                    metavar='lambda1', help='dropout ratio')
parser.add_argument('--share_net', default=2, type=int,
                    metavar='share', help='[1,2,3,4,5]the start number of shared network in the two-stream networks')
parser.add_argument('--re_rank', default='no', type=str, help='performing reranking. [random_walk | k_reciprocal | no]')
parser.add_argument('--pcb', default='off', type=str, help='performing PCB, on or off')
parser.add_argument('--local_feat_dim', default=2048, type=int,
                    help='feature dimention of each local feature in PCB')

parser.add_argument('--patch_size_h', default=3, type=float, help=' patch size heght')
parser.add_argument('--patch_size_w', default=1, type=float, help=' patch size weight')
parser.add_argument('--frame_size', default=2, type=float, help='frame patch size')

parser.add_argument('--label_smooth', default='on', type=str, help='performing label smooth or not')
parser.add_argument('--w_center', default=2, type=float, help='the weight for center loss')

parser.add_argument('--augc', default=1 , type=int,
                    metavar='aug', help='use channel aug or not')
parser.add_argument('--rande', default= 0.5, type=float,
                    metavar='ra', help='use random erasing or not and the probability')
parser.add_argument('--kl', default= 0.05, type=float,
                    metavar='kl', help='use kl loss and the weight') ##0.05
parser.add_argument('--alpha', default=1 , type=int,
                    metavar='alpha', help='magnification for the hard mining')
parser.add_argument('--gamma', default=1, type=int,
                    metavar='gamma', help='gamma for the hard mining')
parser.add_argument('--square', default= 1 , type=int,
                    metavar='square', help='gamma for the hard mining')

parser.add_argument('--beta', default=1, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=1, type=float,
                    help='cutmix probability')
parser.add_argument('--EPS', type=float, default=1e-5, help='episillon')
parser.add_argument('--tau', type=float, default=1.0, metavar='LR')

parser.add_argument("--inv-coeff", type=float, default=25.0,
                    help='Invariance regularization loss coefficient')
parser.add_argument("--var-coeff", type=float, default=25.0,
                    help='Variance regularization loss coefficient')
parser.add_argument("--cov-coeff", type=float, default=1.0,
                    help='Covariance regularization loss coefficient')


args = parser.parse_args()
# os.environ['CUDA_DEVICE_ORDER'] =','.join(map(str, [0, 3]))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# args = parser.parse_args()
# os.environ['CUDA_DEVICE_ORDER'] ='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#set_seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
dataset = args.dataset

np.random.seed(1)

#添加
seq_lenth = 14
# seq_len = 10
test_batch = 32
data_set = VCM()
log_path = args.log_path + 'VCM_log/'
test_mode = [1, 2]
height = args.img_h
width = args.img_w

 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
# n_class = data_set.num_train_pids
n_class = 500
nquery = data_set.num_query_tracklets

ngall = data_set.num_gallery_tracklets
nquery_1 = data_set.num_query_tracklets_1
ngall_1 = data_set.num_gallery_tracklets_1

print('==> Building model..')
net = embed_net(args.low_dim, n_class, share_net=args.share_net,
                h_size = args.img_h, w_size = args.img_w,
                patch_size_h = args.patch_size_h, patch_size_w = args.patch_size_w,
                frame = seq_lenth,
                frame_size = args.frame_size, pcb=args.pcb, local_feat_dim=args.local_feat_dim ,
                drop=args.drop, part=args.part, arch=args.arch, wpa=args.wpa)
net.to(device)    
cudnn.benchmark = True

print('==> Resuming from checkpoint..')
print('model name:  {}'.format(args.resume))

checkpoint_path = args.model_path
if len(args.resume)>0:   
    model_path = checkpoint_path + args.resume
    # model_path = checkpoint_path + 'test_best.t'
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        # pdb.set_trace()
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}!!!!!!!!!!'.format(args.resume))


# if args.method =='id':
criterion = nn.CrossEntropyLoss()
criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.Pad(10),

    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

if dataset == 'VCM':
    rgb_pos, ir_pos = GenIdx(data_set.rgb_label, data_set.ir_label)
queryloader = DataLoader(
    VideoDataset_test(data_set.query, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)

galleryloader = DataLoader(
    VideoDataset_test(data_set.gallery, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)

queryloader_1 = DataLoader(
    VideoDataset_test(data_set.query_1, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)

galleryloader_1 = DataLoader(
    VideoDataset_test(data_set.gallery_1, seq_len=seq_lenth, sample='video_test', transform=transform_test),
    batch_size=test_batch, shuffle=False, num_workers=args.workers)

print('Data Loading Time:\t {:.3f}'.format(time.time()-end))

feature_dim = 2048
if args.arch =='resnet50':
    pool_dim = 2048
elif args.arch =='resnet18':
    pool_dim = 512


def test1():
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0

    if args.pcb == 'on':
        feat_dim = 6 * args.local_feat_dim
    else:
        feat_dim = 2048

    gall_feat = np.zeros((ngall, feat_dim))
    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            input = imgs
            label = pids
            batch_num = input.size(0)

            input = Variable(input.cuda())
            feat = net(input, input, test_mode[0], seq_len=seq_lenth)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num

            g_pids.extend(pids)
            g_camids.extend(camids)

    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, feat_dim))

    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            input = imgs
            label = pids

            batch_num = input.size(0)

            input = Variable(input.cuda())
            feat = net(input, input, test_mode[1], seq_len=seq_lenth)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()

            ptr = ptr + batch_num

            q_pids.extend(pids)
            q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity



    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    print("Computing CMC and mAP")
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)

    ranks = [1, 5, 10, 20]
    print("Results -----------------------------------------------------------------")
    # print("testmAP: {:.2%}".format(mAP))
    # print("testmAP: {:.2%}".format(mAP_k))
    print("CMC curve")
    print("------------------------Matmul-------------------------------------------")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    return cmc, mAP


def test2():
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')

    x_test = []
    y_test = []

    start = time.time()
    ptr = 0
    if args.pcb == 'on':
        feat_dim = 6 * args.local_feat_dim
    else:
        feat_dim = 2048
    gall_feat = np.zeros((ngall_1, feat_dim))

    q_pids, q_camids = [], []
    g_pids, g_camids = [], []
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader_1):
            input = imgs
            input = Variable(input.cuda())
            label = pids
            batch_num = input.size(0)

            feat = net(input, input, test_mode[1], seq_len=seq_lenth)


            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
            #
            g_pids.extend(pids)
            g_camids.extend(camids)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery_1, feat_dim))
    with torch.no_grad():
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader_1):
            input = imgs
            label = pids
            batch_num = input.size(0)
            input = Variable(input.cuda())

            feat = net(input, input, test_mode[0], seq_len=seq_lenth)

            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num

            q_pids.extend(pids)
            q_camids.extend(camids)

    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()


    ###---------------------------------------------------------------------------------------
    # compute the similarity

    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    ranks = [1, 5, 10, 20]
    print("Results -----------------------------------------------------------------")
    print("testmAP: {:.2%}".format(mAP))
    # print("testmAP: {:2%}".format(mAP_k))
    print("CMC curve")
    print("------------------------Matmul-------------------------------------------")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    print("-------------------------------------------------------------------------")
    return cmc, mAP

# testing

if 'v2t' in args.resume:
    cmc_1, mAP_1 = test2()
    print('rgb2ir:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
        cmc_1[0], cmc_1[4], cmc_1[9], cmc_1[19], mAP_1))

if 't2v' in args.resume:
    cmc, mAP = test1()
    print('ir2rgb:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP))

