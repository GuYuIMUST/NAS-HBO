import sys
import time
import numpy as np
import torch
import utils
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from master import transforms as transforms
from master.dataloader import lunanod
import os
from models.cnn_res import *
import logging
import pandas as pd

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--epochs', type=int, default=1200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')  # 20
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

# origin Lu
parser.add_argument('--fold_validate', type=int, default=3, help="fold for validate")
parser.add_argument('--fold_test', type=int, default=5, help="fold for test")


# parser.add_argument('--test_flag', type=bool, default=False, help="Continuing Run")
parser.add_argument('--test_flag', type=bool, default=True, help="Continuing Run")
parser.add_argument('--pretrain_model_path', type=str, default=r"D:\wangmansheng\wangmansheng_2\code\NAS-Lung-master-PC_DARTS_3D\PC-DARTS-master\eval-EXP-20230905-090242\82.52427006693719_57_model.pt", help='pretrain_model_path')

args = parser.parse_args()


CROPSIZE = 32
gbtdepth = 1
fold_validate = args.fold_validate
fold_test = args.fold_test
blklst = []

args.save = 'test-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

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


transform_validate = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((pixmean), (pixstd)),
])

# load data list
trfnamelst = []
trlabellst = []
trfeatlst = []

valfnamelst = []
vallabellst = []
valfeatlst = []

tesfnamelst = []
teslabellst = []
tesfeatlst = []


dataframe = pd.read_csv('../data/annotationdetclsconvfnl_v3.csv',
                        names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])

alllst = dataframe['seriesuid'].tolist()[1:]
labellst = dataframe['malignant'].tolist()[1:]
crdxlst = dataframe['coordX'].tolist()[1:]
crdylst = dataframe['coordY'].tolist()[1:]
crdzlst = dataframe['coordZ'].tolist()[1:]
dimlst = dataframe['diameter_mm'].tolist()[1:]

# validate id
validate_idlst = []
for fname in os.listdir(r'D:\wangmansheng\data\LUNA\rowfile\subset' + str(fold_validate) + '/'):

    if fname.endswith('.mhd'):
        validate_idlst.append(fname[:-4])

# test id
test_idlst = []
for fname in os.listdir(r'D:\wangmansheng\data\LUNA\rowfile\subset' + str(fold_test) + '/'):

    if fname.endswith('.mhd'):
        test_idlst.append(fname[:-4])

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
    # feat = np.hstack((np.reshape(data, (-1,)) / 255, float(d)))
    y, x, z = np.ogrid[-CROPSIZE / 2:CROPSIZE / 2, -CROPSIZE / 2:CROPSIZE / 2, -CROPSIZE / 2:CROPSIZE / 2]
    mask = abs(y ** 3 + x ** 3 + z ** 3) <= abs(float(d)) ** 3
    feat = np.zeros((CROPSIZE, CROPSIZE, CROPSIZE), dtype=float)
    feat[mask] = 1

    if srsid.split('-')[0] in validate_idlst:
        valfnamelst.append(srsid + '.npy')
        vallabellst.append(int(label))
        valfeatlst.append(feat)
    elif srsid.split('-')[0] in test_idlst:
        tesfnamelst.append(srsid + '.npy')
        teslabellst.append(int(label))
        tesfeatlst.append(feat)
    else:
        trfnamelst.append(srsid + '.npy')
        trlabellst.append(int(label))
        trfeatlst.append(feat)
for idx in range(len(trfeatlst)):
    # trfeatlst[idx][0] /= mxx
    # trfeatlst[idx][1] /= mxy
    # trfeatlst[idx][2] /= mxz
    trfeatlst[idx][-1] /= mxd
for idx in range(len(valfeatlst)):
    # tefeatlst[idx][0] /= mxx
    # tefeatlst[idx][1] /= mxy
    # tefeatlst[idx][2] /= mxz
    valfeatlst[idx][-1] /= mxd
for idx in range(len(tesfeatlst)):
    # tefeatlst[idx][0] /= mxx
    # tefeatlst[idx][1] /= mxy
    # tefeatlst[idx][2] /= mxz
    valfeatlst[idx][-1] /= mxd


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    # torch.manual_seed(args.seed)
    cudnn.enabled = True
    # torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)


    genotype = eval("genotypes.%s" % args.arch)
    pretrain_model_path = args.pretrain_model_path
    checkpoint = torch.load(pretrain_model_path)
    model = checkpoint['model']
    model = model.cuda()
    epoch = checkpoint['epoch']
    valid_acc = checkpoint['valid_acc']
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    testdateset = lunanod(preprocesspath, tesfnamelst, teslabellst, tesfeatlst, train=False, download=True, transform=transform_test)
    test_queue = torch.utils.data.DataLoader(testdateset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc %f valid_acc %f epoch %d', test_acc, valid_acc, epoch)


def infer(test_queue, model, criterion):
    with torch.no_grad():
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()

        correct = 0
        total = 0
        TP = FP = FN = TN = 0

        for step, (input, target, feat) in enumerate(test_queue):
            with torch.no_grad():
                input = Variable(input).cuda()
                target = Variable(target).cuda()

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()

            TP += ((predicted == 1) & (target.data == 1)).cpu().sum()
            TN += ((predicted == 0) & (target.data == 0)).cpu().sum()
            FN += ((predicted == 0) & (target.data == 1)).cpu().sum()
            FP += ((predicted == 1) & (target.data == 0)).cpu().sum()

        tpr = 100. * TP.data.item() / (TP.data.item() + FN.data.item())
        fpr = 100. * FP.data.item() / (FP.data.item() + TN.data.item())
        Accuracy = 100. * (TP.data.item() + TN.data.item())/ (TP.data.item() + FN.data.item() + TN.data.item() + FP.data.item())
        Sensitivity = 100. * TP.data.item() / (TP.data.item() + FN.data.item())
        Specificity = 100. * TN.data.item() / (FP.data.item() + TN.data.item())
        F1_Score = 100. * (2. * TP.data.item()) / (2. * TP.data.item() + FP.data.item() + FN.data.item())

        logging.info('tpr ' + str(tpr) + ' fpr ' + str(fpr))
        logging.info('Accuracy ' + str(Accuracy) + ' Sensitivity ' + str(Sensitivity) + ' Specificity ' + str(Specificity) + ' F1_Score ' + str(F1_Score))
        logging.info('test %03d %e %f', step, objs.avg, top1.avg)
        return top1.avg, objs.avg


if __name__ == '__main__':
    main()
