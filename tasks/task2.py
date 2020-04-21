# %%

import sklearn
import cv2 as cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from util import draw_bboxes


# %%

def bgr_to_rgb(image):
    b, g, r = cv2.split(image)
    return cv2.merge([r, g, b])


def draw_rect(img, box):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)





def get_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

def get_center_velocity(track_info):
    return track_info[4], track_info[5]


def check_overlap_xy(box, proposal):
    b_x1, b_y1, b_x2, b_y2 = box
    p_x1, p_y1, p_x2, p_y2 = proposal

    x_outside = p_x2 < b_x1 or p_x1 > b_x2
    y_outside = p_y2 < b_y1 or p_y1 > b_y2

    x_inside = p_x2 < b_x2 and p_x1 > b_x1
    y_inside = p_y2 < b_y2 and p_y1 > b_y1

    return x_outside, y_outside, x_inside, y_inside


def is_leaving(box_center, proposal_center, velocity):
    direction = velocity * (proposal_center - box_center)
    return direction > 0


def tracking_box(img_seq, proposals_seq, instances_seq, tracking_seq, box):
    # box: [x, y, w, h]
    output_seq = []
    for i in range(len(img_seq)):
        img = img_seq[i].copy()
        proposals_list = proposals_seq[i]
        instances_list = instances_seq[i]
        tracking_list = tracking_seq[i]

        n_pedestrians_inside = 0
        n_pedestrians_enter = 0
        n_pedestrians_leave = 0

        draw_rect(img, box)
        img = draw_bboxes(img, proposals_list, instances_list)
        for j in range(len(proposals_list)):
            proposal = proposals_list[j]
            instance = instances_list[j]
            track_info = tracking_list[j]
            x_outside, y_outside, x_inside, y_inside = check_overlap_xy(box, proposal)
            # print((x_outside, y_outside, x_inside, y_inside))

            if x_outside or y_outside:
                pass
            elif x_inside and y_inside:
                n_pedestrians_inside += 1
            else:
                box_cx, box_cy = get_center(box)
                proposal_cx, proposal_cy = get_center(proposal)
                proposal_vx, proposal_vy = get_center_velocity(track_info)
                if not x_inside and y_inside:
                    if abs(proposal_vx) > abs(proposal_vy):
                        b_c, p_c, p_v = box_cx, proposal_cx, proposal_vx
                    else:
                        b_c, p_c, p_v = box_cy, proposal_cy, proposal_vy
                elif not x_inside:
                    b_c, p_c, p_v = box_cx, proposal_cx, proposal_vx
                elif not y_inside:
                    b_c, p_c, p_v = box_cy, proposal_cy, proposal_vy

                if is_leaving(b_c, p_c, p_v):
                    n_pedestrians_leave += 1
                else:
                    n_pedestrians_enter += 1


        text = "Inside: " + str(n_pedestrians_inside) + "  Entering: " + str(n_pedestrians_enter) + "  Leaving: " + str(
            n_pedestrians_leave)
        cv2.putText(img, text, (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 2)
        output_seq.append(img)

    return output_seq
