from __future__ import print_function
import time
time_start = time.time()

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
import shutil
from torch.utils.data import DataLoader

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
    parser.add_argument('--batch_size', type=int, default=1,
                        help='')
    parser.add_argument('--frame_last', type=int, default=100000,
                        help='')

    return parser.parse_args()



def infer(net, img_batch, thresh, cuda, shrink):

    with torch.no_grad():

        img_batch = Variable(img_batch)
        if cuda:
            img_batch = img_batch.cuda()
        y = net(img_batch)      # forward pass
        detections = y.data

        # scale each detection back up to the image
        scale = torch.Tensor([ img_batch.shape[3]/shrink, img_batch.shape[2]/shrink,
                             img_batch.shape[3]/shrink, img_batch.shape[2]/shrink] )

        det_batch = []

        for b in range(img_batch.size(0)):
            det = []
            for i in range(detections.size(1)):
                j = 0
                while detections[b, i, j, 0] >= thresh:
                    score = detections[b, i, j, 0].cpu().numpy()
                    #label_name = labelmap[i-1]
                    pt = (detections[b, i, j, 1:]*scale).cpu().numpy()
                    coords = (pt[0], pt[1], pt[2], pt[3])
                    det.append([pt[0], pt[1], pt[2], pt[3], score])
                    j += 1
            if (len(det)) == 0:
                det = [ [0.1,0.1,0.2,0.2,0.01] ]
            det = np.array(det)
            keep_index = np.where(det[:, 4] >= 0)[0]
            det = det[keep_index, :]
            det_batch.append(det)

        return det_batch



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

    thresh = 0.01

    img = cv2.imread('../detector-mayi/test/sample_mid01/inputs/00000.jpg')
    [img_h, img_w] = [img.shape[0], img.shape[1]]
    shrink_const = 400.0
    max_im_shrink = ( (shrink_const*shrink_const) / (img_h * img_w)) ** 0.5
    shrink = max_im_shrink if max_im_shrink < 1 else 1

    dataloader = DataLoader(ImageFolder('../detector-mayi/test/sample_mid01/inputs', shrink), batch_size=opt.batch_size, shuffle=False, num_workers=8)

    time_detect_start_0 = time.time()

    for batch_idx, (img_og_batch, img_batch) in enumerate(dataloader):

        ## `break` if there are no frames left
        if batch_idx == len(dataloader)-1:
            break

        ## Log progress
        if batch_idx % opt.log_step_batch == 0:
            time_detect_start = time.time()

        img_og_batch = img_og_batch.numpy()

        det_batch = infer(net, img_batch, thresh, opt.cuda, shrink)

        for b in range(opt.batch_size):
            frame_idx = batch_idx * opt.batch_size + b

            img_og = img_og_batch[b]
            det = det_batch[b]

            inds = np.where(det[:, -1] >= opt.visual_threshold)[0]
            if len(inds) != 0:
                for i in inds:
                    bbox = det[i, :4]
                    score = det[i, -1]
                    cv2.rectangle(img_og, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0,0,255], 3)
                    cv2.putText(img_og, '%.3f' % score, (int(bbox[0]), int(bbox[1])), 0, 1, [0,0,255], 3)
            cv2.imwrite(opt.save_folder+'%05d.jpg' % (frame_idx), img_og)

            ## Log progress
            if (batch_idx+1) % opt.log_step_batch == 0 and b == opt.batch_size-1:
                print('#### FPS {:5.2f} -- face-detect #{:4} - #{:4}'
                    .format(opt.log_step_batch * opt.batch_size / (time.time()-time_detect_start), frame_idx - opt.log_step_batch * opt.batch_size + 1, frame_idx))

        if frame_idx > opt.frame_last:
            break

    # Log progress
    if (frame_idx+1) % (opt.log_step_batch*opt.batch_size) != 0:
        print('#### FPS {:5.2f} -- face-detect #{:4} - #{:4}'
            .format((frame_idx % (opt.log_step_batch*opt.batch_size) + 1)/(time.time()-time_detect_start), frame_idx - frame_idx % (opt.log_step_batch*opt.batch_size), frame_idx))
    # Log progress - total
    print('##   FPS {:5.2f} -- face-detect #{:4} - #{:4}'
        .format((frame_idx+1)/(time.time()-time_detect_start_0), 0, frame_idx))



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
            print('#### FPS {:5.2f} -- f2v #{:4} - #{:4}'
                .format(opt.log_step/(time.time()-time_f2v_start), i-opt.log_step+1, i))

    ## Log progress
    if (i+1) % opt.log_step != 0:
        print('#### FPS {:5.2f} -- f2v #{:4} - #{:4}'
            .format((i % opt.log_step + 1)/(time.time()-time_f2v_start), i - i % opt.log_step, i))

    out.release()



if __name__ == '__main__':

    opt = parse_args()

    opt.log_step_batch = int(opt.log_step / opt.batch_size)

    if opt.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # detect (img_folder -> img_folder)
    print('## {:8.2f} sec -- detect start'.format(time.time()-time_start))
    detect_frames()
    print('## {:8.2f} sec -- detect complete'.format(time.time()-time_start))

    # to_video (img_folder -> video)
    print('## {:8.2f} sec -- frame_to_video start'.format(time.time()-time_start))
    frame_to_video()
    print('## {:8.2f} sec -- frame_to_video complete'.format(time.time()-time_start))
