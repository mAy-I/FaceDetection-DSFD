from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torch.utils.data as data
import pdb
import numpy as np
import cv2
import math
import time
import shutil
from torch.utils.data import DataLoader

from data import TestBaseTransform
from face_ssd import build_ssd
from utils.datasets import *



def parse_args():

    parser = argparse.ArgumentParser(description='DSFD:Dual Shot Face Detector')

    parser.add_argument('--trained_model', default='weights/WIDERFace_DSFD_RES152.pth', type=str,
                        help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='eval_tools/', type=str,
                        help='Dir to save results')
    parser.add_argument('--visual_threshold', default=0.1, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use cuda to train model')
    parser.add_argument('--log_step', type=int, default=50,
                        help='')

    return parser.parse_args()



def infer(net, img, transform, thresh, cuda, shrink):
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    with torch.no_grad():
        x = Variable(x.unsqueeze(0))
        if cuda:
            x = x.cuda()
        # print (shrink, x.shape)
        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([ img.shape[1]/shrink, img.shape[0]/shrink,
                             img.shape[1]/shrink, img.shape[0]/shrink] )
        det = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                score = detections[0, i, j, 0].cpu().numpy()
                #label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                det.append([pt[0], pt[1], pt[2], pt[3], score])
                j += 1
        if (len(det)) == 0:
            det = [ [0.1,0.1,0.2,0.2,0.01] ]
        det = np.array(det)

        keep_index = np.where(det[:, 4] >= 0)[0]
        det = det[keep_index, :]
        return det



def detect_frames():

    if os.path.exists(opt.save_folder):
        shutil.rmtree(opt.save_folder)
    os.mkdir(opt.save_folder)

    # load net
    num_classes = 2 # face, background => 2
    min_dim = 640
    net = build_ssd('test', min_dim, num_classes) # initialize SSD
    net.load_state_dict(torch.load(opt.trained_model))
    net.cuda()
    net.eval()
    print('Finished loading model!')

    # evaluation
    transform = TestBaseTransform((104, 117, 123))
    thresh = 0.01

    img = cv2.imread('../detector-mayi/test/sample_mid01/inputs/00000.jpg')
    [img_h, img_w] = [img.shape[0], img.shape[1]]
    shrink_const = 400.0
    max_im_shrink = ( (shrink_const*shrink_const) / (img_h * img_w)) ** 0.5
    shrink = max_im_shrink if max_im_shrink < 1 else 1
    # print('max_im_shrink: ' + str(max_im_shrink))
    # print('shrink: ' + str(shrink))

    dataloader = DataLoader(ImageFolder('../detector-mayi/test/sample_mid01/inputs', shrink), batch_size=1, shuffle=False, num_workers=8)

    for frame_idx, (img_og_batch, img_batch) in enumerate(dataloader):

        ## Log progress
        if frame_idx % opt.log_step == 0:
            time_visualize_start = time.time()

        img_batch = img_batch.squeeze()
        img = img_batch.numpy()
        img_og_batch = img_og_batch.squeeze()
        img_og = img_og_batch.numpy()

        det = infer(net, img, transform, thresh, opt.cuda, shrink)

        inds = np.where(det[:, -1] >= opt.visual_threshold)[0]
        if len(inds) != 0:
            for i in inds:
                bbox = det[i, :4]
                score = det[i, -1]
                cv2.rectangle(img_og, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0,0,255], 3)
                cv2.putText(img_og, '%.3f' % score, (int(bbox[0]), int(bbox[1])), 0, 1, [0,0,255], 3)
        cv2.imwrite(opt.save_folder+'%05d.jpg' % (frame_idx), img_og)

        ## Log progress
        if (frame_idx+1) % opt.log_step == 0:
            print('#### FPS {:4.2f} -- visualize #{:4} - #{:4}'
                .format(opt.log_step/(time.time()-time_visualize_start), frame_idx-opt.log_step+1, frame_idx))



def frame_to_video():

    opt = parse_args()

    if os.path.isfile('face.mp4'):
        os.remove('face.mp4')

    files = [f for f in os.listdir('eval_tools/') if os.path.isfile(os.path.join('eval_tools/', f))]
    ## For sorting the file name properly
    files.sort(key = lambda x: int(x[:-4]))

    img = cv2.imread('eval_tools/' + files[0])
    out = cv2.VideoWriter('face.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (img.shape[1], img.shape[0]))

    for i in range(len(files)):

        ## Log progress
        if i % opt.log_step == 0:
            time_f2v_start = time.time()

        filename = 'eval_tools/' + files[i]
        ## Reading each files
        img = cv2.imread(filename)
        out.write(img)

        ## Log progress
        if (i+1) % opt.log_step == 0:
            print('#### FPS {:4.1f} -- f2v #{:4} - #{:4}'
                .format(opt.log_step/(time.time()-time_f2v_start), i-opt.log_step+1, i))

    ## Log progress
    if (i+1) % opt.log_step != 0:
        print('#### FPS {:4.1f} -- f2v #{:4} - #{:4}'
            .format((i % opt.log_step + 1)/(time.time()-time_f2v_start), i - i % opt.log_step, i))

    out.release()



if __name__ == '__main__':

    opt = parse_args()

    if opt.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    detect_frames()

    # frame_to_video()
