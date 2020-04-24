# %%

import torch
import cv2
import numpy as np
from net.csp import Csp
from utils.nms.py_cpu_nms import py_cpu_nms
from config import Config
from load_data.load_data import get_pets2009
import os
from collections import OrderedDict



# %%

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    new_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        # print(k)
        if k.split('.')[0] != 'module':
            k = 'module.' + k
        new_dict[k] = v
        # print(k)
    try:
        model.load_state_dict(pretrained_dict)
    except:
        model.load_state_dict(new_dict)
    return model


def parse_det_offset(pos, scale, offset, config):
    size, score, down, nms_thresh = config.test_size, config.score_thres, config.stride, config.nms_thres
    height = scale[0, :, :]
    width = scale[1, :, :]

    offset_y = offset[0, :, :]
    offset_x = offset[1, :, :]
    y_c, x_c = np.where(pos > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            # w = 0.41 * h
            w = np.exp(width[y_c[i], x_c[i]]) * down
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = pos[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, size[1]), min(y1 + h, size[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = py_cpu_nms(boxs, nms_thresh)
        boxs = boxs[keep, :]
    return boxs


def test(return_groundtruth = False):
    config = Config()
    torch.set_grad_enabled(False)
    # Initialize CSP network
    net = Csp('test')
    use_cuda = config.use_cuda
    print('Use cuda: ' + str(use_cuda))
    net = load_model(net, config.csp_checkpoint, not use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")
    net = net.to(device)
    net = torch.nn.DataParallel(net, device_ids=list(range(config.num_gpu)))
    net.eval()
    # Get testset
    _, _, _, test_img, test_label, test_mean = get_pets2009(dir='./data_PETS2009', pkl_dir='./data_cache')
    img_sequence = test_img.copy()
    test_img = np.float32(test_img)
    test_img -= test_mean
    test_img = test_img.transpose(0, 3, 1, 2)

    start = 0
    testset_size = test_img.shape[0]

    boxes = []

    while start < testset_size:
        # Release GPU memory
        torch.cuda.empty_cache()
        batch_start = start
        batch_end = min(start + config.test_batch_size, testset_size)
        test_batch = torch.from_numpy(test_img[batch_start:batch_end])
        test_batch = test_batch.to(device)

        pos, scale, offset = net(test_batch)
        pos = pos.data.cpu().numpy()
        scale = scale.data.cpu().numpy()
        offset = offset.data.cpu().numpy()
        for i in range(batch_end - batch_start):
            curr_boxes = parse_det_offset(pos[i], scale[i], offset[i], config)
            boxes.append(curr_boxes)
        start += config.test_batch_size
    if return_groundtruth:
        return img_sequence, boxes, test_label
    return img_sequence, boxes


def test_all_datasets(checkpoints):
    config = Config()
    torch.set_grad_enabled(False)
    # Initialize CSP network
    use_cuda = config.use_cuda
    print('Use cuda: ' + str(use_cuda))

    # Get testset (768 imgs
    _, _, _, test_img_whole, test_label_whole, test_mean_whole = get_pets2009(dir='./data_PETS2009',
                                                                              pkl_dir='./data_cache', cache=True)
    img_sequence_whole = test_img_whole.copy()
    test_img_whole = np.float32(test_img_whole)
    test_img_whole -= test_mean_whole
    test_img_whole = test_img_whole.transpose(0, 3, 1, 2)
    print(test_img_whole.shape)
    dataset_begin = 0
    dataset_end = 0
    boxes = []

    for checkpoint in checkpoints:
        net = Csp('test')
        net = load_model(net, os.path.join('./checkpoints/CSP_Crossvalidation', checkpoint + '.pth'), not use_cuda)
        dataset_begin = dataset_end
        dataset_end = int(checkpoint)
        test_img = test_img_whole[dataset_begin:dataset_end]
        # test_label = test_label_whole[dataset_begin, dataset_end]
        device = torch.device("cuda" if use_cuda else "cpu")
        net = net.to(device)
        net = torch.nn.DataParallel(net, device_ids=list(range(config.num_gpu)))
        net.eval()

        start = 0
        testset_size = test_img.shape[0]

        while start < testset_size:
            # Release GPU memory
            torch.cuda.empty_cache()
            batch_start = start
            batch_end = min(start + config.test_batch_size, testset_size)
            test_batch = torch.from_numpy(test_img[batch_start:batch_end])
            test_batch = test_batch.to(device)

            pos, scale, offset = net(test_batch)
            pos = pos.data.cpu().numpy()
            scale = scale.data.cpu().numpy()
            offset = offset.data.cpu().numpy()
            for i in range(batch_end - batch_start):
                curr_boxes = parse_det_offset(pos[i], scale[i], offset[i], config)
                boxes.append(curr_boxes)
            start += config.test_batch_size
    return img_sequence_whole, boxes
