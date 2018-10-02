import argparse
import os
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torchvision.datasets as datasets
from models import get_model
# standard torchvision transforms library
import torchvision.transforms as transforms
# local transforms
import transforms as mytransforms
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch Multitask Augmentation')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--arch', default='wideresnet', type=str,
                    help='network architecture')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--milestones', default=[60, 120, 160], nargs='+', type=int,
                    help='learning rate step decay positions')
parser.add_argument('--gamma', default=0.2, type=int,
                    help='learning rate decay factor')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True,
                    type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10-12', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_false')
parser.add_argument('--num-rotate-classes', default=12, type=int,
                    help="number of rotation classes")
parser.add_argument('--rotation-prob', default=0.25, type=float,
                    help="Probabiity of doing rotation")
parser.add_argument('--rotation-range', default=360, type=float,
                    help="Range of possible rotation degree, defualt is full range(360)")
parser.add_argument('--alpha', default=0.5, type=float,
                    help="factor of loss from rotated images")
parser.set_defaults(augment=True)


best_prec1 = 0

random.seed(0)
np.random.seed(0)


def main():
    global args, best_prec1
    args = parser.parse_args()
    # torch.cuda.set_device(args.gpu)
    if args.tensorboard:
        print("Using TensorBoard")
        configure("exp/%s" % (args.name))

    # Data loading code
    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='edge'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('../data', train=True, download=True,
                                                transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()](
            '../data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    # create model
    model = get_model(args.arch, args.dataset, args.num_rotate_classes)
    model = model.cuda()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define learning rate scheduler
    if not args.milestones:
        milestones = [args.epochs]
    else:
        milestones = args.milestones
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.gamma, last_epoch=args.start_epoch - 1)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: {}'.format(best_prec1))


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_aux = AverageMeter()
    top1_cls = AverageMeter()
    top1_aux = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if random.uniform(0, 1) <= args.rotation_prob:
            input, target_aux = mytransforms.AuxRandomRotation(
                input,  args.num_rotate_classes)
            target = target.cuda(async=True)
            input = input.cuda()
            target_aux = target_aux.cuda(async=True)
            cls, aux = model(input)

            loss_cls = criterion(cls, target)
            loss_aux = criterion(aux, target_aux)

            loss = args.alpha * loss_cls + loss_aux

            # measure accuracy and record loss
            prec_cls = accuracy(cls.data, target, topk=(1,))[0]
            prec_aux = accuracy(aux.data, target_aux, topk=(1,))[0]

            losses_cls.update(loss_cls.item(), input.size(0))
            losses_aux.update(loss_aux.item(), input.size(0))
            losses.update(loss.item(), input.size(0))

            top1_cls.update(prec_cls, input.size(0))
            top1_aux.update(prec_aux, input.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:4f} ({loss.avg:.4f})\t'
                      'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                      'Loss_ {loss_aux.val:.4f} ({loss_aux.avg:.4f})\t'
                      'Prec@1 classification: {top1_cls.val:.3f} ({top1_cls.avg:.3f}) '
                      'Auxiliary: {top1_aux.val:.3f} ({top1_aux.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          loss=losses, loss_cls=losses_cls, loss_aux=losses_aux,
                          top1_cls=top1_cls, top1_aux=top1_aux))
        else:
            target = target.cuda(async=True)
            input = input.cuda()
            cls, _ = model(input)
            # normal output layer
            loss_cls = criterion(cls, target)
            # measure accuracy and record loss
            prec1 = accuracy(cls.data, target, topk=(1,))[0]
            losses_cls.update(loss_cls.item(), input.size(0))
            top1_cls.update(prec1, input.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                      'Prec@1 Classification: {top1_cls.val:.3f} ({top1_cls.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          loss_cls=losses_cls, top1_cls=top1_cls))

    # log to TensorBoard
    if args.tensorboard:
        lr = optimizer.param_groups[0]['lr']
        log_value('learning_rate', lr, epoch)
        log_value('train_loss', losses.avg, epoch)
        log_value('train_loss_cls', losses_cls.avg, epoch)
        log_value('train_acc', top1_cls.avg, epoch)
        log_value('train_loss_aux', losses_aux.avg, epoch)
        log_value('train_acc_aux', top1_aux.avg, epoch)


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input = input.cuda()

            # compute output
            # only the normal output is used in validation
            output, _ = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "exp/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'exp/%s/' %
                        (args.name) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
