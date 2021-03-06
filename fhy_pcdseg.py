import open3d
import argparse
import os
import time
import json
import h5py
import datetime
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from matplotlib import pyplot as plt
import my_log as log
import matplotlib.pyplot as plt

from model.fhy_pointnet1 import PointNetSeg, feature_transform_reguliarzer
from model.pointnet2 import PointNet2SemSeg
from model.qyz_mlp import Net
from model.utils import load_pointnet

from pcd_utils import mkdir, select_avaliable
#from data_utils.SemKITTI_Loader import SemKITTI_Loader
#from data_utils.kitti_utils import getpcd
from data_utils.fhy4_Sem_Loader import SemKITTI_Loader, pcd_normalize
from data_utils.fhy4_datautils_test import Semantic_KITTI_Utils

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/img_velo_label_327-002"

def parse_args(notebook = False):
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--mode', default='diff', choices=('train', 'eval'))
    parser.add_argument('--model_name', type=str, default='pointnet', choices=('pointnet', 'pointnet2'))
    parser.add_argument('--pn2', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')#16
    parser.add_argument('--subset', type=str, default='inview', choices=('inview', 'all'))
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=80, help='number of epochs for training')#60
    parser.add_argument('--pretrain', type=str, default='/media/qing/DATA3/Learning-Based-Terrain-Traversability-Analysis-master/experiment/pointnet/pointnet-0.40758-0022.pth', help='whether use pretrain model')
    parser.add_argument('--linear_pretrain', type=str,default='/media/qing/DATA3/Learning-Based-Terrain-Traversability-Analysis-master/experiment/MLP_linear/MLP_linear-0.08614054-0001.pth',help='whether use pretrain model')
    parser.add_argument('--angular_pretrain', type=str,default='/media/qing/DATA3/Learning-Based-Terrain-Traversability-Analysis-master/experiment/MLP_angular/MLP_angular-0.02084015-0001.pth',help='whether use pretrain model')
    #parser.add_argument('--pretrain', type=str, default=None, help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')#0.001
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--augment', default=False, action='store_true', help="Enable data augmentation")
    if notebook:
        #if using in jupyter notebook, you should change ' ' to '[]'
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    if args.pn2 == False:
        args.model_name = 'pointnet'
    else:
        args.model_name = 'pointnet2'
    return args

def calc_decay(init_lr, epoch):
    return init_lr * 1 / (1 + 0.03 * epoch)#0.03
    #return init_lr

def test_kitti_semseg(model, loader, model_name, num_classes, class_names):
    ious = np.zeros((num_classes,), dtype = np.float32)
    count = np.zeros((num_classes,), dtype = np.uint32) 
    count[0] = 1
    accuracy = []
    
    
    for points, target in tqdm(loader, total=len(loader), smoothing=0.9, dynamic_ncols=True):
    # in tqdm, loader is an iterable variable. loader here includes dataset with label
        batch_size, num_point, _ = points.size() #points
        points = points.float().transpose(2, 1).cuda()
        target = target.long().cuda()

        with torch.no_grad():
            if model_name == 'pointnet':
                pred, _ = model(points)
            if model_name == 'pointnet2':
                pred,_ = model(points)

            pred_choice = pred.argmax(-1)
            target = target.squeeze(-1)

            for class_id in range(num_classes):
                I = torch.sum((pred_choice == class_id) & (target == class_id)).cpu().item()
                U = torch.sum((pred_choice == class_id) | (target == class_id)).cpu().item()
                iou = 1 if U == 0 else I/U
                ious[class_id] += iou
                count[class_id] += 1

            correct = (pred_choice == target).sum().cpu().item()
            accuracy.append(correct/ (batch_size * num_point)) 

    categorical_iou = ious / count
    df = pd.DataFrame(categorical_iou, columns=['mIOU'], index=class_names)
    df = df.sort_values(by='mIOU', ascending=False)


    log.info('categorical mIOU')
    log.msg(df)

    acc = np.mean(accuracy)
    miou = np.mean(categorical_iou[1:])
    return acc, miou

def train(args):
    experiment_dir = mkdir('experiment/')
    checkpoints_dir = mkdir('experiment/%s/'%(args.model_name))
    
    kitti_utils = Semantic_KITTI_Utils(ROOT, subset = args.subset)
    class_names = kitti_utils.class_names
    num_classes = kitti_utils.num_classes

    if args.subset == 'inview':
        train_npts = 2000#2000
        test_npts = 2500#2500
    
    if args.subset == 'all':
        train_npts = 10000#50000
        test_npts = 12500#100000

    log.info(subset=args.subset, train_npts=train_npts, test_npts=test_npts)

    dataset = SemKITTI_Loader(ROOT, train_npts, train = True, subset = args.subset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.workers, pin_memory=True)
    
    test_dataset = SemKITTI_Loader(ROOT, test_npts, train = False, subset = args.subset)
    testdataloader = DataLoader(test_dataset, batch_size=int(args.batch_size/2), shuffle=False, 
                            num_workers=args.workers, pin_memory=True)

    if args.model_name == 'pointnet':
        model = PointNetSeg(num_classes, input_dims = 4, feature_transform = True)
    if args.model_name == 'pointnet2':
        #model = PointNet2SemSeg(num_classes, feature_dims = 1)
        model = PointNet2SemSeg(num_classes)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4)

    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)
    #use more than 1 gpu
    model.cuda()
    log.info('Using gpu:',args.gpu)
    
    if args.pretrain is not None:
        log.info('Use pretrain model...')
        model.load_state_dict(torch.load(args.pretrain))
        init_epoch = int(args.pretrain[:-4].split('-')[-1])
        #init_epoch = 0
        log.info('Restart training', epoch=init_epoch)
    else:
        log.msg('Training from scratch')
        init_epoch = 0

    best_acc = 0
    best_miou = 0

    #->5.12 add
    # to show details of epoch training
    loss_list = []
    miou_list = []
    acc_list = []
    t_miou_list = []
    t_acc_list = []
    epoch_time = []
    lr_list = []
    #-<5.12 add

    for epoch in range(init_epoch,args.epoch):
        model.train()
        lr = calc_decay(args.learning_rate, epoch)
        log.info(model=args.model_name, gpu=args.gpu, epoch=epoch, lr=lr)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        for points, target in tqdm(dataloader, total=len(dataloader), smoothing=0.9, dynamic_ncols=True):
            points = points.float().transpose(2, 1).cuda()
            target = target.long().cuda()

            if args.model_name == 'pointnet':
                logits, trans_feat = model(points)
            if args.model_name == 'pointnet2':
                logits, _ = model(points)

            #logits = logits.contiguous().view(-1, num_classes)
            #target = target.view(-1, 1)[:, 0]
            #loss = F.nll_loss(logits, target)

            logits = logits.transpose(2, 1)
            loss = nn.CrossEntropyLoss()(logits, target)

            if args.model_name == 'pointnet':
                loss += feature_transform_reguliarzer(trans_feat) * 0.001

          #  loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.cuda.empty_cache()
        '''train acc'''
        t_acc, t_miou = test_kitti_semseg(model.eval(), dataloader,
                                    args.model_name,num_classes,class_names)

        acc, miou = test_kitti_semseg(model.eval(), testdataloader,
                                    args.model_name,num_classes,class_names)

        # miou_list.append(np.asscalar(miou))
        # acc_list.append(np.asscalar(acc))
        save_model = False
        if acc > best_acc:
            best_acc = acc
        
        if miou > best_miou:
            best_miou = miou
            save_model = True
        
        #->5.12 add
        loss_list.append(loss.item())
        t_miou_list.append(np.asscalar(t_miou))
        t_acc_list.append(np.asscalar(t_acc))
        miou_list.append(np.asscalar(miou))
        acc_list.append(np.asscalar(acc))
        epoch_time.append(epoch)
        lr_list.append(lr)
        #->5.12 add
        if save_model:
            fn_pth = '%s-%.5f-%04d.pth' % (args.model_name, best_miou, epoch)
            log.info('Save model...',fn = fn_pth)
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, fn_pth))
        else:
            log.info('No need to save model')
        # 3.31 add |>
        # show(args)
        # 3.31 add |<
        log.warn('train_Curr',accuracy=t_acc, mIOU=t_miou)
        log.warn('Curr',accuracy=acc, mIOU=miou)
        log.warn('Best',accuracy=best_acc, mIOU=best_miou)
    # 5.15 add
    label_size = {"size":10}
    fig = plt.figure()
    plt.plot(epoch_time,loss_list,label = "loss")
    plt.plot(epoch_time,t_miou_list,label = "t_mIOU")
    plt.plot(epoch_time,t_acc_list,label = "t_accuracy")
    plt.plot(epoch_time,miou_list,label = "mIOU")
    plt.plot(epoch_time,acc_list,label = "accuracy")
   # plt.plot(epoch_time,lr_list,label = "learning rate")
    plt.xlabel("epoch time", fontsize=10)
    plt.ylabel("value", fontsize=10)
    plt.title("training trendency", fontsize=20)
    plt.tick_params(labelsize=10)
    plt.legend(prop = label_size)
    plt.show()

def show(args):
    kitti_utils = Semantic_KITTI_Utils(ROOT, subset = args.subset)
    pth_path = ROOT + "/experiment/pointnet"
    pths = os.listdir(pth_path)
    pths.sort()
    pth_new = os.path.join(pth_path, pths[-1])
    print(pth_new)
    model = load_pointnet(args.model_name, kitti_utils.num_classes, pth_new)
    part = '03'
    index = 607
    points, labels = kitti_utils.get_pts_l(part, index, True)
    pts3d = points[:,:-1]
    pcd = pcd_normalize(points)
    points_tensor = torch.from_numpy(pcd).unsqueeze(0).transpose(2, 1).float().cuda()
    with torch.no_grad():
        logits,_ = model(points_tensor)
        pred = logits[0].argmax(-1).cpu().numpy()
    pts2d = kitti_utils.project_3d_to_2d(pts3d)
    pred_color = np.ndarray.tolist(kitti_utils.mini_color_BGR[pred])
    orig_color = np.ndarray.tolist(kitti_utils.mini_color_BGR[predlabels])
    img1 = kitti_utils.draw_2d_points(pts2d, orig_color)
    img2 = kitti_utils.draw_2d_points(pts2d, pred_color)
    img = np.hstack((img1, img2))
    cv2.imshow('img',img)
    cv2.waitKey(0)

def evaluate(args):
    kitti_utils = Semantic_KITTI_Utils(ROOT, subset = args.subset)
    class_names = kitti_utils.class_names
    num_classes = kitti_utils.num_classes

    if args.subset == 'inview':
        test_npts = 2000
    
    if args.subset == 'all':
        test_npts = 100000

    test_dataset = SemKITTI_Loader(ROOT, test_npts, train=False, subset=args.subset)
    testdataloader = DataLoader(test_dataset, batch_size=int(args.batch_size/2), shuffle=False, num_workers=args.workers)

    model = load_pointnet(args.model_name, kitti_utils.num_classes, args.pretrain)

    acc, miou = test_kitti_semseg(model.eval(), testdataloader,args.model_name,num_classes,class_names)

    log.info('Curr', accuracy=acc, mIOU=miou)

def produce_data(model, loader, model_name, num_classes, class_names):
    data = torch.tensor([])
    data = data.cuda()
    for points, target in tqdm(loader, total=len(loader), smoothing=0.9, dynamic_ncols=True):
    # in tqdm, loader is an iterable variable. loader here includes dataset with label
        batch_size, num_point, _ = points.size() #points
        points = points.float().transpose(2, 1).cuda()
        target = target.long().cuda()

        with torch.no_grad():
            if model_name == 'pointnet':
                pred, _ = model(points)
            if model_name == 'pointnet2':
                pred,_ = model(points)

            pred_choice = pred.argmax(-1)
            target = target.squeeze(-1)
            pred_choice = pred_choice.unsqueeze(1)
            new_points = torch.cat((points,pred_choice),1)
            #print(pred_choice.shape)
            #print(new_points.shape)
            data = torch.cat((data,new_points),0)


    return pred_choice,target,data

def produce_vel(model, loader):
    for points in tqdm(loader, total=len(loader), smoothing=0.9, dynamic_ncols=True):
        points = points.float().transpose(2, 1).cuda()
        with torch.no_grad():
            prediction = model(points)
            print(prediction)



def diffusion(args):
    kitti_utils = Semantic_KITTI_Utils(ROOT, subset=args.subset)
    class_names = kitti_utils.class_names
    num_classes = kitti_utils.num_classes

    if args.subset == 'inview':
        test_npts = 2000

    if args.subset == 'all':
        test_npts = 100000

    test_dataset = SemKITTI_Loader(ROOT, test_npts, train=False, subset=args.subset)
    testdataloader = DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=False,
                                num_workers=args.workers)

    seg_model = load_pointnet(args.model_name, kitti_utils.num_classes, args.pretrain)
    label,target,data= produce_data(seg_model.eval(), testdataloader,args.model_name,num_classes,class_names)
    target = target.squeeze(-1)
    torch.set_printoptions(profile='full')
    print(data.shape)

    #new_testdataloader = DataLoader(data,batch_size=1,shuffle=False,num_workers=args.workers)

    LinearVel_model = Net(5, 64)
    AngularVel_model = Net(5, 64)
    LinearVel_model = torch.nn.DataParallel(LinearVel_model)
    AngularVel_model = torch.nn.DataParallel(AngularVel_model)
    torch.backends.cudnn.benchmark = True

    # use more than 1 gpu
    log.info('Using gpu:', '0')
    LinearVel_model.load_state_dict(torch.load(args.linear_pretrain))
    AngularVel_model.load_state_dict(torch.load(args.angular_pretrain))

    LinearVel_model.cuda()
    LinearVel_model.eval()
    AngularVel_model.cuda()
    AngularVel_model.eval()

    with torch.no_grad():
        for points in data:
            points = points.unsqueeze(0)
            #print(points.shape)
            prediction_linear = LinearVel_model(points)
            prediction_angular = AngularVel_model(points)
            prediction = torch.cat((prediction_linear,prediction_angular))
            prediction.resize(2,1)
            print(prediction)








if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.mode == "train":
        train(args)
    if args.mode == "eval":
        evaluate(args)
    if args.mode == "diff":
        diffusion(args)


