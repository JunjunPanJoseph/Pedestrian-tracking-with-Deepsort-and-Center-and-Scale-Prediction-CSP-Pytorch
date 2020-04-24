import numpy as np
import cv2 as cv2
import os
import xml.dom.minidom
import pickle


def get_citypersons(root_dir='data/cityperson', type='train'):
    image_data = []
    image_set_path = os.path.join(root_dir, 'ImageSets', type + '.txt')
    fid_set = open(image_set_path)
    while True:
        image_name = fid_set.readline().replace('\n', '')
        if len(image_name) == 0:
            break
        fid_label = open(os.path.join(root_dir, 'Annotations', image_name[:-3] + 'txt'))
        boxes = []
        while True:
            box_info = fid_label.readline().replace('\n', '')
            if len(box_info) == 0:
                break
            box_info = box_info.split(' ')
            box = [int(box_info[x]) for x in range(len(box_info))]
            boxes.append(box)

        annotation = {}
        annotation['filepath'] = os.path.join(root_dir, 'Images', image_name)
        annotation['bboxes'] = np.array(boxes)
        image_data.append(annotation)

    return image_data


def get_image_sequence(dir):
    images = os.listdir(dir)
    return_list = []
    for img in images:
        img = os.path.join(dir, img)
        t = cv2.imread(img)
        return_list.append(t)
    return np.array(return_list)


def get_boxes_sequence(dir):
    dom = xml.dom.minidom.parse(dir)
    root = dom.documentElement
    frames = root.getElementsByTagName('frame')
    labels = []
    for frame in frames:
        frame_no = frame.getAttribute("number")
        objects = frame.getElementsByTagName('object')
        labels_curr = []
        for obj in objects:
            id = obj.getAttribute("id")
            # One object only have one box
            box = obj.getElementsByTagName("box")[0]
            xc = float(box.getAttribute("xc"))
            yc = float(box.getAttribute("yc"))
            w = float(box.getAttribute("w"))
            h = float(box.getAttribute("h"))
            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            # labels_curr.append(np.array([x, y, w, h, id]))
            labels_curr.append(np.array([x1, y1, x2, y2]))
        labels.append(np.array(labels_curr))
    return labels


def _get_pets2009(dir='./data_PETS2009'):
    sub_folders = os.listdir(dir)
    train_img = []
    test_img = []
    train_label = []
    test_label = []
    for folder in sub_folders:
        sequence_path = os.path.join(dir, folder, 'View_001')
        xml_path = os.path.join(dir, folder, 'label.xml')
        images_sequence = get_image_sequence(sequence_path)
        labels_sequence = get_boxes_sequence(xml_path)
        if images_sequence.shape[0] != len(labels_sequence):
            print(sequence_path)
            print('# img = ' + str(images_sequence.shape[0]))
            print('# labels = ' + str(len(labels_sequence)))
            raise Exception('# img != # labels')
        print('Load sequence: ' + str(images_sequence.shape))
        if folder == 'S2L1':
            # Use first 150 images for testing
            start = 0
            end = 150
            # train_img.append(images_sequence[:start])
            # train_label += labels_sequence[:start]
            test_img.append(images_sequence[start:end])
            test_label += labels_sequence[start:end]
            train_img.append(images_sequence[end:])
            train_label += labels_sequence[end:]
        else:
            train_img.append(images_sequence)
            train_label += labels_sequence
    train_img = np.concatenate(train_img, axis=0)
    test_img = np.concatenate(test_img, axis=0)
    train_mean = np.mean(train_img, axis=(0, 1, 2))
    test_mean = np.mean(test_img, axis=(0, 1, 2))
    return train_img, train_label, train_mean, test_img, test_label, test_mean


# def get_pets2009(dir = '../data_PETS2009', pkl_dir='../data_cache'):
def get_pets2009(dir, pkl_dir, cache=False):
    path = os.path.join(pkl_dir, 'PETS2009.npz')
    # Load cache files
    if not cache:
        # load from dir
        output_files = _get_pets2009(dir)
        train_img, train_label, train_mean, test_img, test_label, test_mean = output_files
        return train_img, train_label, train_mean, test_img, test_label, test_mean
    try:
        with np.load(path, allow_pickle=True) as dataset:
            train_img = dataset['train_img']
            train_label = dataset['train_label']
            train_mean = dataset['train_mean']
            test_img = dataset['test_img']
            test_label = dataset['test_label']
            test_mean = dataset['test_mean']
            return train_img, train_label, train_mean, test_img, test_label, test_mean
    except:
        # load from dir
        output_files = _get_pets2009(dir)
        train_img, train_label, train_mean, test_img, test_label, test_mean = output_files
        np.savez(path, train_img=train_img, train_label=train_label, train_mean=train_mean,
                 test_img=test_img, test_label=test_label, test_mean=test_mean)
        return train_img, train_label, train_mean, test_img, test_label, test_mean

