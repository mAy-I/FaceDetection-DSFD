from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import WIDERFace_ROOT , WIDERFace_CLASSES as labelmap
from PIL import Image
from data import WIDERFaceDetection, WIDERFaceAnnotationTransform, WIDERFace_CLASSES, WIDERFace_ROOT, BaseTransform , TestBaseTransform
from data import *
import torch.utils.data as data
from face_ssd import build_ssd
#from resnet50_ssd import build_sfd
import pdb
import numpy as np
import cv2
import math
import time
import shutil



def parse_args():

    parser = argparse.ArgumentParser(description='DSFD:Dual Shot Face Detector')

    parser.add_argument('--trained_model', default='weights/WIDERFace_DSFD_RES152.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='eval_tools/', type=str,
                        help='Dir to save results')
    parser.add_argument('--visual_threshold', default=0.1, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use cuda to train model')
    parser.add_argument('--widerface_root', default=WIDERFace_ROOT, help='Location of WIDERFACE root directory')

    return parser.parse_args()



def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[0:750, :]
    return dets



def infer(net , img , transform , thresh , cuda , shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    with torch.no_grad():
        x = Variable(x.unsqueeze(0))
        # x = Variable(x.unsqueeze(0) , volatile=True)
        if cuda:
            x = x.cuda()
        #print (shrink , x.shape)
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



def infer_flip(net , img , transform , thresh , cuda , shrink):
    img = cv2.flip(img, 1)
    det = infer(net , img , transform , thresh , cuda , shrink)
    det_t = np.zeros(det.shape)
    det_t[:, 0] = img.shape[1] - det[:, 2]
    det_t[:, 1] = det[:, 1]
    det_t[:, 2] = img.shape[1] - det[:, 0]
    det_t[:, 3] = det[:, 3]
    det_t[:, 4] = det[:, 4]
    return det_t



def detect_frames():

    # load net
    cfg = widerface_640
    num_classes = len(WIDERFace_CLASSES) + 1 # +1 background
    net = build_ssd('test', cfg['min_dim'], num_classes) # initialize SSD
    net.load_state_dict(torch.load(opt.trained_model))
    net.cuda()
    net.eval()
    print('Finished loading model!')

    # evaluation
    cuda = opt.cuda
    transform = TestBaseTransform((104, 117, 123))
    thresh = cfg['conf_thresh']

    factor = 2

    for frame_idx in range(0, 1200):

        ## Log progress
        if frame_idx % 10 == 0:
            time_visualize_start = time.time()

        img = cv2.imread('../detector-mayi/test/sample_mid01/inputs/%05d.jpg' % (frame_idx), cv2.IMREAD_COLOR)

        max_im_shrink = ( (2000.0*2000.0) / (img.shape[0] * img.shape[1])) ** 0.5
        shrink = max_im_shrink if max_im_shrink < 1 else 1

        det0 = infer(net , img , transform , thresh , cuda , shrink)
        det1 = infer_flip(net , img , transform , thresh , cuda , shrink)
        # shrink detecting and shrink only detect big face
        st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
        det_s = infer(net , img , transform , thresh , cuda , st)
        index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
        det_s = det_s[index, :]
        # enlarge one times
        bt = min(factor, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
        det_b = infer(net , img , transform , thresh , cuda , bt)
        # enlarge small iamge x times for small face
        if max_im_shrink > factor:
            bt *= factor
            while bt < max_im_shrink:
                det_b = np.row_stack((det_b, infer(net , img , transform , thresh , cuda , bt)))
                bt *= factor
            det_b = np.row_stack((det_b, infer(net , img , transform , thresh , cuda , max_im_shrink) ))
        # enlarge only detect small face
        if bt > 1:
            index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
            det_b = det_b[index, :]
        else:
            index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
            det_b = det_b[index, :]
        det = np.row_stack((det0, det1, det_s, det_b))
        det = bbox_vote(det)

        inds = np.where(det[:, -1] >= opt.visual_threshold)[0]
        if len(inds) != 0:
            for i in inds:
                bbox = det[i, :4]
                score = det[i, -1]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [0,0,255], 3)
                cv2.putText(img, '%.3f' % score, (int(bbox[0]), int(bbox[1])), 0, 1, [0,0,255], 3)
        cv2.imwrite(opt.save_folder+'%05d.jpg' % (frame_idx), img)

        ## Log progress
        if (frame_idx+1) % 10 == 0:
            print('#### FPS {:4.1f} -- visualize #{:4} - #{:4}'
                .format(10/(time.time()-time_visualize_start), frame_idx-10+1, frame_idx))



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
        if i % 50 == 0:
            time_f2v_start = time.time()

        filename = 'eval_tools/' + files[i]
        ## Reading each files
        img = cv2.imread(filename)
        out.write(img)

        ## Log progress
        if (i+1) % 50 == 0:
            print('#### FPS {:4.1f} -- f2v #{:4} - #{:4}'
                .format(50/(time.time()-time_f2v_start), i-50+1, i))

    ## Log progress
    if (i+1) % 50 != 0:
        print('#### FPS {:4.1f} -- f2v #{:4} - #{:4}'
            .format((i % 50 + 1)/(time.time()-time_f2v_start), i - i % 50, i))

    out.release()



if __name__ == '__main__':

    opt = parse_args()

    if opt.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if os.path.exists(opt.save_folder):
        shutil.rmtree(opt.save_folder)
    os.mkdir(opt.save_folder)

    detect_frames()
    # frame_to_video()