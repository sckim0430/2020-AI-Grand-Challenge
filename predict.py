"""2020 AI Grand Challenge Predict Src
"""
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import create_dataloader
from models.experimental import attempt_load
import numpy as np
import torch
from pathlib import Path
import os
import sys
import json

base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, base_path)


def iou_matching(bbox, candidates):
    """IOU Matching

    Args:
        bbox (np.ndarray): gt bbox coordinates array
        candidates (np.ndarray): predict bbox coordinates array

    Returns:
        int, float: Candidate Bbox index, IOU
    """
    if candidates.size == 0:
        return -1, -1

    bbox_tl, bbox_br = bbox[:2], bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = (bbox_br-bbox_tl).prod()
    area_candidates = (candidates_br - candidates_tl).prod(axis=1)

    iou = area_intersection / (area_bbox + area_candidates - area_intersection)

    candidates_index = np.argmax(iou)
    iou = np.max(iou)

    return candidates_index, iou


def main(string):
    #set parameter
    collapse_check_list = [148, 149, 150]
    model_path = os.path.join(base_path, 'yolov5l_19_0.988_0.764.pt')
    model_path2 = os.path.join(base_path, 'yolov5l_34_0.988_0.794.pt')
    model_path3 = os.path.join(base_path, 'yolov5l_34_0.989_0.769.pt')

    json_path = os.path.join(base_path, 't1_res_U0000000279.json')

    img_size = 640
    confidence_threshold = 0.175  # 0.1, 0.2
    iou_threshold = 0.4  # 0.4, 0.5
    tracking_iou = 0.65
    device_choice = ''
    fps_collapse_rate = 150
    batch_size = 20
    single_cls = True

    device = select_device(device_choice)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(model_path, map_location=device)  # load FP32 model
    model2 = attempt_load(model_path2, map_location=device)  # load FP32 model
    model3 = attempt_load(model_path3, map_location=device)  # load FP32 model
    img_size = check_img_size(img_size, s=model.stride.max())  # check img_size
    model.half()
    model2.half()
    model3.half()
    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    # run once
    _ = model2(img.half() if half else img) if device.type != 'cpu' else None
    # run once
    _ = model3(img.half() if half else img) if device.type != 'cpu' else None

    with open(json_path, 'w') as f:
        json_file = dict()
        json_file['annotations'] = []
        json.dump(json_file, f)

    f.close()

    for folder_path in os.listdir(string):

        if os.path.splitext(folder_path)[-1] == '.cache':
            continue

        folder_path = os.path.join(string, folder_path)

        info_of_video = []
        Answer_Manager_list = []

        dataloader = create_dataloader(folder_path, img_size, batch_size, model.stride.max(
        ), single_cls, pad=0.5, rect=False)[0]

        for batch_i, (img, _, paths, shapes) in enumerate(dataloader):
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = img.shape  # batch size, channels, height, width

            with torch.no_grad():
                # Run model
                # inference and training outputs
                inf_out, _ = model(img, augment=True)
                inf_out2, _ = model2(img, augment=True)
                inf_out3, _ = model3(img, augment=True)

                inf_out = torch.cat([inf_out, inf_out2, inf_out3], 1)

                for index, _ in enumerate(inf_out):
                    sort, idx = torch.sort(_[:, 4], 0, descending=True)

                    inf_out[index] = _[idx]

                # Run NMS
                output = non_max_suppression(
                    inf_out, conf_thres=confidence_threshold, iou_thres=iou_threshold)

            for si, pred in enumerate(output):
                path = Path(paths[si])
                pred_boxes = np.asarray([])
                pred_scores = np.asarray([])

                if len(pred):
                    pred[:, :4] = scale_coords(
                        img[si].shape[1:], pred[:, :4], shapes[si][0], shapes[si][1])

                    pred_boxes = np.asarray(pred[:, :4].tolist())
                    pred_scores = np.asarray(pred[:, 4].tolist())

                    delete_index = np.where(np.logical_or(abs(
                        pred_boxes[:, 0]-pred_boxes[:, 2]) <= 32, abs(pred_boxes[:, 1]-pred_boxes[:, 3]) <= 32))

                    pred_scores = np.delete(pred_scores, delete_index, axis=0)
                    pred_boxes = np.delete(pred_boxes, delete_index, axis=0)

                info_of_frame = dict()
                info_of_frame['file_name'] = os.path.basename(str(path))
                info_of_frame['boxes'] = pred_boxes.astype('int')
                info_of_frame['scores'] = pred_scores

                info_of_video.append(info_of_frame)

        #tracking
        for index, info in enumerate(info_of_video):

            info_boxes = info['boxes']
            info_scores = info['scores']

            info_of_video[index]['boxes'] = []
            info_of_video[index]['scores'] = []

            #정답 Manager 유무 확인
            if len(Answer_Manager_list) != 0:

                #pred 유무 확인
                if len(info_scores) != 0:

                    delete_index_list = []

                    for idx, answer in enumerate(Answer_Manager_list):
                        delete_index, iou = iou_matching(
                            answer['box'], info_boxes)

                        if iou >= tracking_iou:
                            Answer_Manager_list[idx]['box'] = info_boxes[delete_index]
                            Answer_Manager_list[idx]['score'] = info_scores[delete_index]
                            Answer_Manager_list[idx]['stack'] += 1

                            info_of_video[index]['boxes'].append(
                                info_boxes[delete_index].tolist())
                            info_of_video[index]['scores'].append(
                                info_scores[delete_index].tolist())

                            # del info_boxes[delete_index]
                            # del info_scores[delete_index]
                            info_boxes = np.delete(
                                info_boxes, [delete_index], axis=0)
                            info_scores = np.delete(
                                info_scores, [delete_index], axis=0)

                        else:

                            if answer['stack'] >= fps_collapse_rate:
                                delete_index_list.append(idx)

                            else:
                                Answer_Manager_list[idx]['stack'] += 1
                                info_of_video[index]['boxes'].append(
                                    answer['box'].tolist())
                                info_of_video[index]['scores'].append(
                                    answer['score'].tolist())

                    Answer_Manager_list = np.delete(
                        Answer_Manager_list, delete_index_list)
                    Answer_Manager_list = Answer_Manager_list.tolist()

                    if index < len(info_of_video)-fps_collapse_rate:
                        for info_box, info_score in zip(info_boxes[:], info_scores):
                            for check_val in collapse_check_list:
                                _, iou = iou_matching(info_box, np.array(
                                    info_of_video[index+check_val]['boxes']))

                                if iou >= tracking_iou:
                                    answer_dict = dict()
                                    answer_dict['box'] = info_box
                                    answer_dict['score'] = info_score
                                    answer_dict['stack'] = 1
                                    Answer_Manager_list.append(answer_dict)

                                    info_of_video[index]['boxes'].append(
                                        info_box.tolist())
                                    info_of_video[index]['scores'].append(
                                        info_score.tolist())

                                    break

                else:
                    delete_index_list = []

                    for idx, answer in enumerate(Answer_Manager_list):
                        if answer['stack'] >= fps_collapse_rate:
                            delete_index_list.append(idx)
                        else:
                            Answer_Manager_list[idx]['stack'] += 1
                            info_of_video[index]['boxes'].append(
                                answer['box'].tolist())
                            info_of_video[index]['scores'].append(
                                answer['score'].tolist())

                    Answer_Manager_list = np.delete(
                        Answer_Manager_list, delete_index_list)
                    Answer_Manager_list = Answer_Manager_list.tolist()

            else:

                #pred 유무 확인
                if len(info_scores) != 0:
                    if index < len(info_of_video)-fps_collapse_rate:
                        for info_box, info_score in zip(info_boxes[:], info_scores):
                            for check_val in collapse_check_list:
                                _, iou = iou_matching(info_box, np.array(
                                    info_of_video[index+check_val]['boxes']))

                                if iou >= tracking_iou:
                                    answer_dict = dict()
                                    answer_dict['box'] = info_box
                                    answer_dict['score'] = info_score
                                    answer_dict['stack'] = 1
                                    Answer_Manager_list.append(answer_dict)

                                    info_of_video[index]['boxes'].append(
                                        info_box.tolist())
                                    info_of_video[index]['scores'].append(
                                        info_score.tolist())

                                    break

        with open(json_path, 'r') as f:
            json_data = json.load(f)

        f.close()

        for _ in info_of_video:
            detection_num = len(_['scores'])

            # if detection_num == 0:
            #     continue

            json_d = dict()
            json_d['file_name'] = _['file_name']
            json_d['box'] = []

            for index in range(detection_num):
                j_d = dict()
                j_d['position'] = _['boxes'][index]
                j_d['confidence_score'] = str(_['scores'][index])
                json_d['box'].append(j_d)

            json_data['annotations'].append(json_d)

        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        f.close()


if __name__ == "__main__":
    main(sys.argv[1])
