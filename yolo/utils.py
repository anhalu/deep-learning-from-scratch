from collections import Counter
import torch


# import numpy as np


def intersection_over_union(box_pred, box_label, box_format='corners'):
    """
    convert all box coordinate to corner format:
    """
    global pred_x1, pred_y1, pred_x2, pred_y2, label_x1, label_y1, label_x2, label_y2
    if box_format == 'mid_point':  # xywh
        pred_x1 = box_pred[0] - box_pred[2] / 2
        pred_y1 = box_pred[1] - box_pred[3] / 2
        pred_x2 = box_pred[0] + box_pred[2] / 2
        pred_y2 = box_pred[1] + box_pred[3] / 2

        label_x1 = box_label[0] - box_label[2] / 2
        label_y1 = box_label[1] - box_label[3] / 2
        label_x2 = box_label[0] + box_label[2] / 2
        label_y2 = box_label[1] + box_label[3] / 2

    if box_format == 'corners':  # xyxy
        pred_x1 = box_pred[0]
        pred_y1 = box_pred[1]
        pred_x2 = box_pred[3]
        pred_y2 = box_pred[4]

        label_x1 = box_label[0]
        label_y1 = box_label[1]
        label_x2 = box_label[3]
        label_y2 = box_label[4]

    intersection_x1 = max(pred_x1, label_x1)
    intersection_y1 = max(pred_y1, label_y1)
    intersection_x2 = min(pred_x2, label_x2)
    intersection_y2 = min(pred_y2, label_y2)
    intersection_area = max(intersection_x2 - intersection_x1, 0) * max(intersection_y2 - intersection_y1, 0)
    return intersection_area / ((pred_x2 - pred_x1) * (pred_y2 - pred_y1) - intersection_area)


def non_max_suppression(boxes, prob_threshold, iou_threshold, box_format='corners'):
    """
    Phương pháp triệt tiêu box trùng nhau trên cùng 1 object
    - Nếu prob của bbox đó nhỏ hơn 1 ngưỡng nào đó thì bỏ qua luôn
    - với tất cả các box còn lại :
        + Lấy bbox có prob lớn nhất
        + xóa tất cả các box có độ IOU (so với box đã chọn ở trên) > nghưỡng (threshold : vd 0.5)

    boxes = [[class_idx, conf, x1, y2, x2, y2], [..], ...] if box_format == 'cornets'


    note :
    - khi compare thì nó áp dụng từng clas riêng biệt một.
    - Khi mà iou > ngưỡng, tức là có khả năng 2 box đó đang dự đoán cùng 1 vật thể -> remove box có đang xét.
             iou < ngưỡng : tức là 2 box đó khả năng dự đoán là 2 object khác nhau.

    """
    assert boxes == list
    boxes = [box for box in boxes if box[1] > prob_threshold]
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    boxes_after_nms = []

    while boxes:
        chosen_box = boxes.pop(0)
        boxes_after_nms.append(chosen_box)
        boxes = [
            box for box in boxes
            if box[1] != chosen_box[1] or intersection_over_union(box, chosen_box) < iou_threshold
        ]
    return boxes_after_nms


def mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format='corners', num_classes=20
):
    """
    - mean average precision = mean of average precision for all classes
    if iou of predicted vs target box > iou_threshold -> True Positive
    else -> False Positive

    False Negative : didn't output bbox for a target box
    True Negative : no meaning : không suất ra bb nơi mà không có object ở đó.

    P : Trong tất cả số prediction bbox thì số lượng bbox dự đoán chính xác.

    R : trong tổng số target bbox, thì bao nhiêu box mô hình đã dự đoán là chính xác.
        (chia cho số target bbox).

    step :
        1. Get all bounding box predictions on out test set
        2. Sort by descending confidence score
        3. Calculate the precision and Recall as we go through all outputs (quan trọng phân tích tiếp dưới)
        4. Plot the Precision-Recall graph
        5. Calculate Area under PR curve
        6. This was only for one class, we need to calculate for all classes.
            mAP = (AP_class1 + AP_class2 + ... + AP_classN) / N

        7. All this was calculated given a specific IoU threshold of 0.5, we need to redo all
            computations for many IoUs, example: 0.5, 0.55, 0.6, ... 0.95. Then average this (trung bình của nó)
            this will be out final result. This is what is meant by mAP@0.5:0.005:0.95


    - Quan trọng cần chú ý : Với mỗi IoU sẽ cho ra 1 PR curve khác nhau
        + Đối với toàn bộ dataset: Đối với tất cả các detections được đự đoán ra dựa trên IoU sẽ tính ra TP hoặc FP.
        + Sau đó sort lại tất cả các detection theo giảm dần confident score. Ta sẽ tính các accFP và accTP là các giá trị tích lũy
        của TP và FP
    -> Sau đó tính ap theo 11 point interpolation hoặc interpolation all point
    """

    # pred_box = [[train_idx, class_prob, prob_score, x1, y1, x2, y2], ...] , target same

    average_precision = []
    epsilon = 1e-6  # ổn định giá trị hàm số.

    for c in range(num_classes):
        detections = [detection for detection in pred_boxes if detection[1] == c]
        ground_truths = [gt for gt in target_boxes if gt[1] == c]

        # img 0 có 3 target box
        # img 1 có 5 target box
        # amount boxes = {0: 3, 1: 5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            all_gt_img = [gt for gt in ground_truths if
                          detection[0] == gt[0]]  # get all target box in image same with detection train idx

            num_gts = len(ground_truths)
            best_iou = 0
            best_gt_idx = 0
            for idx, gt in enumerate(all_gt_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),  # xyxy
                    torch.tensor(gt[3:])  # xyxy
                )
                if iou > best_iou:  # lấy ra target box có giá trị iou lớn nhất so với detection box
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:  # so sánh xem iou đang xét giữa detection box và target box
                if amount_bboxes[detection[0]][
                    best_gt_idx] == 0:  # kiểm tra xem ở vị trí image thứ detection[0], target idx đã bị detection
                    # nào chiếm chưa
                    TP[detection_idx] = 1  # chưa chiếm thì là true positive
                    amount_bboxes[detection[0]][best_gt_idx] = 1  # update lại thành đã chiếm rồi
                else:
                    FP[detection_idx] = 1  # ngược lại thì là False postive
            else:
                FP[detection_idx] = 1  # ngược lại thì là False postive

        #   [1, 0, 0, 1, 1, 0] -> [1, 1, 1, 2, 2, 2]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        precisions = torch.cat(torch.tensor([1]), precisions)
        recalls = torch.cat(torch.tensor([0]), recalls)

        average_precision.append(torch.trapz(precisions, recalls))

    '''
        -> mean average precision for all class with iou 0.5 -> mAP@0.5 
        -> mAP@0.5:0.05:0.95 -> mean of mAP for all iou from 0.5 -> 0.95 
    '''
    return sum(average_precision) / len(average_precision)
