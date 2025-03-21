import sys
import time
import glob
import numpy as np
import torch
import utils
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from master import transforms as transforms
from master.dataloader import lunanod
import os
import logging
import pandas as pd
from separate_loss import ConvSeparateLoss, TriSeparateLoss

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=7, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--aux_loss_weight', type=float, default=10.0, help='weight decay')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--fold', type=int, default=5, help="fold")
parser.add_argument('--pret', type=int, default=23, help="pret")
parser.add_argument('--sep_loss', type=str, default='l2', help='path to save the model')
args = parser.parse_args()

CROPSIZE = 32
gbtdepth = 1
fold = args.fold
pret = args.pret
blklst = []

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 2

preprocesspath = r'D:\wangmansheng\data\LUNA\cls\crop_v3'

pixvlu, npix = 0, 0
for fname in os.listdir(preprocesspath):
    if fname.endswith('.npy'):
        if fname[:-4] in blklst: continue
        data = np.load(os.path.join(preprocesspath, fname))
        pixvlu += np.sum(data)
        npix += np.prod(data.shape)
pixmean = pixvlu / float(npix)
pixvlu = 0
for fname in os.listdir(preprocesspath):
    if fname.endswith('.npy'):
        if fname[:-4] in blklst: continue
        data = np.load(os.path.join(preprocesspath, fname)) - pixmean
        pixvlu += np.sum(data * data)
pixstd = np.sqrt(pixvlu / float(npix))
print(pixmean, pixstd)
logging.info('mean ' + str(pixmean) + ' std ' + str(pixstd))
# Datatransforms
logging.info('==> Preparing data..')  # Random Crop, Zero out, x z flip, scale,
transform_train = transforms.Compose([
    # transforms.RandomScale(range(28, 38)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomYFlip(),
    transforms.RandomZFlip(),
    transforms.ZeroOut(4),
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),  # need to cal mean and std, revise norm func
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),
])

# load data list
trfnamelst = []
trlabellst = []
trfeatlst = []
tefnamelst = []
telabellst = []
tefeatlst = []


dataframe = pd.read_csv('../data/annotationdetclsconvfnl_v3.csv',
                        names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])

alllst = dataframe['seriesuid'].tolist()[1:]
labellst = dataframe['malignant'].tolist()[1:]
crdxlst = dataframe['coordX'].tolist()[1:]
crdylst = dataframe['coordY'].tolist()[1:]
crdzlst = dataframe['coordZ'].tolist()[1:]
dimlst = dataframe['diameter_mm'].tolist()[1:]
# test id
teidlst = []
for fold_1 in range(fold):
    for fname in os.listdir(r'D:\wangmansheng\data\LUNA\rowfile\subset' + str(fold_1) + '/'):  # change

        if fname.endswith('.mhd'):
            teidlst.append(fname[:-4])
mxx = mxy = mxz = mxd = 0
for srsid, label, x, y, z, d in zip(alllst, labellst, crdxlst, crdylst, crdzlst, dimlst):
    mxx = max(abs(float(x)), mxx)
    mxy = max(abs(float(y)), mxy)
    mxz = max(abs(float(z)), mxz)
    mxd = max(abs(float(d)), mxd)
    if srsid in blklst: continue
    # crop raw pixel as feature
    data = np.load(os.path.join(preprocesspath, srsid + '.npy'))
    bgx = int(data.shape[0] / 2 - CROPSIZE / 2)
    bgy = int(data.shape[1] / 2 - CROPSIZE / 2)
    bgz = int(data.shape[2] / 2 - CROPSIZE / 2)
    data = np.array(data[bgx:bgx + CROPSIZE, bgy:bgy + CROPSIZE, bgz:bgz + CROPSIZE])
    y, x, z = np.ogrid[-CROPSIZE / 2:CROPSIZE / 2, -CROPSIZE / 2:CROPSIZE / 2, -CROPSIZE / 2:CROPSIZE / 2]
    mask = abs(y ** 3 + x ** 3 + z ** 3) <= abs(float(d)) ** 3
    feat = np.zeros((CROPSIZE, CROPSIZE, CROPSIZE), dtype=float)
    feat[mask] = 1
    if srsid.split('-')[0] in teidlst:
        tefnamelst.append(srsid + '.npy')
        telabellst.append(int(label))
        tefeatlst.append(feat)
    else:
        trfnamelst.append(srsid + '.npy')
        trlabellst.append(int(label))
        trfeatlst.append(feat)
for idx in range(len(trfeatlst)):
    # trfeatlst[idx][0] /= mxx
    # trfeatlst[idx][1] /= mxy
    # trfeatlst[idx][2] /= mxz
    trfeatlst[idx][-1] /= mxd
for idx in range(len(tefeatlst)):
    # tefeatlst[idx][0] /= mxx
    # tefeatlst[idx][1] /= mxy
    # tefeatlst[idx][2] /= mxz
    tefeatlst[idx][-1] /= mxd


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion_train = ConvSeparateLoss(weight=args.aux_loss_weight) if args.sep_loss == 'l2' else TriSeparateLoss(
        weight=args.aux_loss_weight)
    criterion_train = criterion_train.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CLASSES, args.layers, criterion, criterion_train)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    trainset = lunanod(preprocesspath, trfnamelst, trlabellst, trfeatlst, train=True, download=True,
                       transform=transform_train)
    train_queue = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=0)  # change,20to0

    testset = lunanod(preprocesspath, tefnamelst, telabellst, tefeatlst, train=False, download=True,
                      transform=transform_test)
    valid_queue = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=0)  # change,20to0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        if epoch >= pret:
            logging.info("alphas_normal = \n{}".format(torch.sigmoid(model.alphas_normal)))
            logging.info("alphas_reduce = \n{}".format(torch.sigmoid(model.alphas_reduce)))
            logging.info("betas_normal = \n{}".format(torch.sigmoid(model.betas_normal)))
            logging.info("betas_reduce = \n{}".format(torch.sigmoid(model.betas_reduce)))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
        logging.info('train_acc %f', train_acc)

        # validation
        if args.epochs - epoch <= 10:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        scheduler.step()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    for step, (input, target, feat) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        input_search, target_search, useless = next(iter(valid_queue))

        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

        if epoch >= pret:
            if args.unrolled:
                architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
            else:
                loss, loss1, loss2 = architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        if step % args.report_freq == 0 and step > 0:
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    with torch.no_grad():
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        model.eval()

        for step, (input, target, feat) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda()
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            if step % args.report_freq == 0 and step > 0:
                logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
