import os
import random
import numpy as np
import pickle
import torch
import cv2
import torch.nn.functional as F

import math

def check_bbox(bbox):
    x_min, y_min, x_max, y_max = max(0, int(bbox[0])),max(0, int(bbox[1])), max(0, int(bbox[2])), max(0, int(bbox[3])) 
    area = (y_max-y_min)*(x_max-x_min)
    new_bbox = [x_min, y_min, x_max, y_max]                                                                                            
    if area > 4:
        return True, new_bbox
    return False, bbox

def check_left(bbox):
    x_min, y_min, x_max, y_max = max(0, int(bbox[0])),max(0, int(bbox[1])), max(0, int(bbox[2])), max(0, int(bbox[3])) 
    area = (y_max-y_min)*(x_max-x_min)
    new_bbox = [x_min, y_min, x_max, y_max]
    if x_min <=2 and y_min <= 2:
        return True, new_bbox
    return False, bbox


def Generate_relation_bbox(Human, Object, new=False, isnp=False):
    if not isnp:
        ans = [0, 0, 0, 0, 0]
        ans[1] = min(Human[0], Object[0])
        ans[2] = min(Human[1], Object[1])
        ans[3] = max(Human[2], Object[2])
        ans[4] = max(Human[3], Object[3])
        res = np.array(ans).reshape(1, 5).astype(np.float64)
        if new:
            return ans
        return res
    else:
        ans = np.zeros_like(Human)
        for i in range(ans.shape[0]):
            ans[(i, 1)] = min(Human[(i, 0)], Object[(i, 0)])
            ans[(i, 2)] = min(Human[(i, 1)], Object[(i, 1)])
            ans[(i, 3)] = min(Human[(i, 2)], Object[(i, 2)])
            ans[(i, 4)] = min(Human[(i, 3)], Object[(i, 3)])
        return ans

def Generate_action_HICO(action_list, num_class=600):
    action_ = np.zeros(num_class)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1, num_class)
    return action_

def Generate_part_score(joint_list):
    score_list_16 = [float(e['score']) for e in joint_list]
    index_list = [0, 1, 4, 5, 6, 9, 10, 12, 13, 15]
    score_list_10 = [max(0.0001, score_list_16[i]) for i in index_list]
    score_list_6 = []
    score_list_6.append((score_list_10[0] + score_list_10[3]) / 2)
    score_list_6.append((score_list_10[1] + score_list_10[2]) / 2)
    score_list_6.append(score_list_10[4])
    score_list_6.append((score_list_10[6] + score_list_10[9]) / 2)
    score_list_6.append((score_list_10[7] + score_list_10[8]) / 2)
    score_list_6.append(score_list_10[5])
    score_list_6 = np.array(score_list_6, dtype=np.float64).reshape((1, 6))
    return score_list_6


def draw_bbox(img, bbox, color=(255, 0, 0)):
    start_point = int(bbox[0]), int(bbox[1])
    end_point = int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, start_point, end_point, color)
    return img

def Generate_action_PVP(idx, num_pvp):
    action_PVP = np.zeros([num_pvp], dtype=np.float64)
    if isinstance(idx, int):
        action_PVP[idx] = 1
    else:
        action_PVP[list(idx)] = 1
    return action_PVP

def Generate_action_object(idx, num_pvp):
    action_PVP = np.zeros([1, num_pvp], dtype=np.float64)
    if isinstance(idx, int) or isinstance(idx, np.int32):
        action_PVP[:, idx] = 1
    else:
        action_PVP[:, list(idx)[0]] = 1
    return action_PVP

def get_infor(ori_ground_truth, index):
    ground_truth = ori_ground_truth
    i = index
    Human = ground_truth[i][2]
    Object = ground_truth[i][3]
    relation = Generate_relation_bbox(Human, Object)
    original_action = ground_truth[i][1]
    action = Generate_action_HICO(ground_truth[i][1])
    part_bbox = ground_truth[i][4]['part_bbox'][None, :, :]
    num_part_local = len(part_bbox[0]) # 10
    action_hp10=Generate_action_PVP(ground_truth[i][4]['hp10_list'], 10)

    gt_object=Generate_action_object(ground_truth[i][4]['object80_list'], 80)
    P_boxes = part_bbox[0]
    PVP0 = Generate_action_PVP(ground_truth[i][4]['pvp76_ankle2'], 12)
    PVP1 = Generate_action_PVP(ground_truth[i][4]['pvp76_knee2'], 10)
    PVP2 = Generate_action_PVP(ground_truth[i][4]['pvp76_hip'], 5)
    PVP3 = Generate_action_PVP(ground_truth[i][4]['pvp76_hand2'], 31)
    PVP4 = Generate_action_PVP(ground_truth[i][4]['pvp76_shoulder2'], 5)
    PVP5 = Generate_action_PVP(ground_truth[i][4]['pvp76_head'], 13)
    P_boxes = np.delete(P_boxes, 0, axis=1)
    return original_action, P_boxes, gt_object, action, PVP0, PVP1, PVP2, PVP3, PVP4, PVP5

def get_infor_ori(ori_ground_truth, index):
    ground_truth = ori_ground_truth
    i = index
    Human_bbox = ground_truth[i][2]
    Object_bbox = ground_truth[i][3]
    relation = Generate_relation_bbox(Human_bbox, Object_bbox)
    original_action = ground_truth[i][1]
    part_bbox = ground_truth[i][4]['part_bbox'][None, :, :]
    num_part_local = len(part_bbox[0]) # 10
    action_hp10=Generate_action_PVP(ground_truth[i][4]['hp10_list'], 10)

    gt_object=ground_truth[i][4]['object80_list']
    P_boxes = part_bbox[0]
    PVP0 = ground_truth[i][4]['pvp76_ankle2']
    PVP1 = ground_truth[i][4]['pvp76_knee2']
    PVP2 = ground_truth[i][4]['pvp76_hip']
    PVP3 = ground_truth[i][4]['pvp76_hand2']
    PVP4 = ground_truth[i][4]['pvp76_shoulder2']
    PVP5 = ground_truth[i][4]['pvp76_head']
    P_boxes = np.delete(P_boxes, 0, axis=1)
    return Human_bbox, Object_bbox, original_action, P_boxes, gt_object, PVP0, PVP1, PVP2, PVP3, PVP4, PVP5


def convert_part_box(box):
    x_min, y_min, x_max, y_max = box
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max) 
    x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), max(0, x_max), max(0, y_max)
    new_bbox = [x_min, y_min, x_max, y_max]
    if (y_max - y_min) > 4 and (x_max - x_min)>4:
        return True, new_bbox
    else:
        return False, new_bbox
