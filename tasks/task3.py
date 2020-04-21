
print_dist = False
import numpy as np
import math
from tasks.task2 import get_center_velocity, get_center
from config import Config
config = Config()
def dot_product(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


def get_dist(dist_list, x, y):
    if x > y:
        x, y = y, x
    return dist_list[x][y]


def set_dist(dist_list, x, y, dist):
    if x > y:
        x, y = y, x
    dist_list[x][y] = dist


def list_to_str(num_list, F=True):
    str = ""
    for n in num_list:
        if F:
            str += " %1.2f" % n
        else:
            str += "    %d" % n
    return str


def show_dist(dist_list, labels):
    print("  " + list_to_str(labels, False))
    for i in range(len(dist_list)):
        print(str(labels[i]) + " " + list_to_str(dist_list[i]))


def box_dist(box1, box2):
    cx1, cy1 = get_center(box1)
    cx2, cy2 = get_center(box2)
    return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

def velocity_dist(state1, state2):
    vx1, vy1 = get_center_velocity(state1)
    vx2, vy2 = get_center_velocity(state2)
    return math.sqrt((vx2 - vx1) ** 2 + (vy2 - vy1) ** 2)

def instances_similarity(box1, state1, box2, state2):
    state_dist = velocity_dist(state1, state2)
    boxes_distance = box_dist(box1, box2)
    a = 1.0
    b = 1 / 7
    final = a * state_dist + b * boxes_distance
    if print_dist:
        print('velocity dist: ' + str(state_dist))
        print('loc dist: ' + str(boxes_distance))
        print('final dist: '+ str(final))
    return final

def labels_to_instances(labels, instances):
    return [instances[i] for i in labels]

def hc(boxes, states, instances, threshold, k = 1):
    dist_list = []
    dist_reference = []
    n_boxes = len(boxes)
    for i in range(n_boxes):
        column = []
        column2 = []
        for j in range(n_boxes):
            if print_dist:
                print('dist between ' + str(instances[i]) + '  ' + str(instances[j]) + ': ')
            similarity = instances_similarity(boxes[i], states[i], boxes[j], states[j])
            column.append(similarity)
            column2.append(similarity)
        dist_list.append(column)
        dist_reference.append(column)

    n_clusters = n_boxes
    clusters_label = list(range(n_boxes))
    while n_clusters > k:
        label_inst = labels_to_instances(clusters_label, instances)
        best_i = 0
        best_j = 0
        smallest_dist = 2147483648
        # max_dist = 0
        for i in range(n_boxes):
            if i not in clusters_label:
                continue
            for j in range(n_boxes):
                if j not in clusters_label:
                    continue
                if i == j:
                    continue
                curr_dist = get_dist(dist_list, i, j)
                if curr_dist < smallest_dist:
                # if curr_dist > max_dist:
                    smallest_dist = curr_dist
                    best_i = i
                    best_j = j
        # merge label i, j
        if smallest_dist > threshold:
            break
        smaller_label = min(best_i, best_j)
        if print_dist:
            show_dist(dist_list, label_inst)
            print("(dist = " + str(smallest_dist) + ")Merge " + str(label_inst[min(best_i, best_j)]) + ", " + str(label_inst[max(best_i, best_j)]) + " ---> " + str(label_inst[smaller_label]))

        for i in range(n_boxes):
            if i not in clusters_label:
                continue
            if i == smaller_label:
                continue
            larger_dist = max(get_dist(dist_list, best_i, i), get_dist(dist_list, best_j, i))
            set_dist(dist_list, smaller_label, i, larger_dist)

        for i in range(n_boxes):
            if clusters_label[i] == best_i or clusters_label[i] == best_j:
                clusters_label[i] = smaller_label

        n_clusters = n_clusters - 1
    cluster = 0
    for i in range(len(clusters_label)):
        exist = False
        for j in range(len(clusters_label)):
            if clusters_label[j] == i:
                exist = True
                clusters_label[j] = cluster
        if exist:
            cluster += 1
    return clusters_label


#%%

def get_group_boxes(boxes):
    x1_,  y1_, x2_, y2_ = [], [], [], []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1_.append(x1)
        y1_.append(y1)
        x2_.append(x2)
        y2_.append(y2)
    return min(x1_), min(y1_), max(x2_), max(y2_)
