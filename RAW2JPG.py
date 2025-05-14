import os
import cv2
import numpy as np
import zlib
import argparse
import json
from ctypes import *
import utils
from concurrent.futures import ThreadPoolExecutor, as_completed

DrawBB = False
Draw3DBB = False
DrawOpenLane = False
DrawCULLINGBB = False
SEGMENTATION = False
SEGMENTATIONRLE = False
COLOR = True
NORMALS = False
BaseColor = False
DEPTH = False
INSTANCE = False
COMPRESSEDRAW = False
VisualFlow = False
SUBFOLDER = ""
THREADCOUNT = 8
import time
import shutil
import datetime
import re

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

class RLE_Header(BigEndianStructure):

    def encode(self):
        return string_at(addressof(self), sizeof(self))

    def decode(self, data):
        memmove(addressof(self), data, sizeof(self))
        return len(data)

    _pack_ = 1
    _fields_ = [
        ('flag', c_ubyte),
        ('rle', c_char * 3),
        ('height', c_ushort),
        ('width', c_ushort)
    ]


class RLE_Data(Structure):

    def encode(self):
        return string_at(addressof(self), sizeof(self))

    def decode(self, data):
        memmove(addressof(self), data, sizeof(self))
        return len(data)

    _pack_ = 1
    _fields_ = [
        ('tag', c_ubyte),
        ('size', c_ushort)
    ]


def decode_rlesegmentation(img, rle_header_):
    rle_header_.decode(img[0:8])
    start_num = 8
    segmentation_data = img[8:]
    filesize = len(segmentation_data)
    index = 0
    segment = np.zeros(HEIGHT * WIDTH, dtype=np.uint8)
    segmentPointer = 0
    while True:
        current_rle_data = RLE_Data()
        current_rle_data.decode(segmentation_data[index:index + 3])
        segment[segmentPointer: segmentPointer + current_rle_data.size] = current_rle_data.tag
        segmentPointer += current_rle_data.size
        index += 3
        if index >= filesize:
            break;
    segment = segment.reshape(HEIGHT, WIDTH)
    return segment


def draw_bbox(img, bbox, color):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    return img


def draw_point(img, point, color):
    cv2.circle(img, (int(point[0]), int(point[1])), 4, color, -1)
    return img


def draw_bboxes(img, bboxes, culledBoxes):
    for key, value in bboxes.items():
        if value > 0:
            img = draw_bbox(img, list(key), [255, 0, 0])
        else:
            img = draw_bbox(img, list(key), [0, 255, 0])
    for key, value in culledBoxes.items():
        if value > 0:
            img = draw_bbox(img, list(key), [255, 0, 0])
        else:
            img = draw_bbox(img, list(key), [0, 0, 255])
    return img


def draw_3DB(img, boxes3DCarHead, boxes3DCarTail):
    pts1 = []
    pts2 = []
    confirm1 = 1
    confirm2 = 1
    for HeadPoint in boxes3DCarHead:
        img = draw_point(img, HeadPoint, [0, 255, 255])
        if (int(HeadPoint[0]) > 0) and (int(HeadPoint[0]) < WIDTH) and (int(HeadPoint[1]) > 0) and (
                int(HeadPoint[1]) < HEIGHT):
            pts1.append((int(HeadPoint[0]), int(HeadPoint[1])))
        else:
            confirm1 = 0
            # print([int(HeadPoint[0]), int(HeadPoint[1])])
    for TailPoint in boxes3DCarTail:
        img = draw_point(img, TailPoint, [255, 255, 0])
        if (int(TailPoint[0]) > 0) and (int(TailPoint[0]) < WIDTH) and (int(TailPoint[1]) > 0) and (
                int(TailPoint[1]) < HEIGHT):
            pts2.append((int(TailPoint[0]), int(TailPoint[1])))
        else:
            confirm2 = 0
            # print([int(TailPoint[0]), int(TailPoint[1])])
    # pts0 = np.array([pts1[0], pts1[1], pts1[2], pts1[3]], np.int32)
    # roi_as.append(pts0.astype(np.int))
    # cv2.polylines(img,roi_as,True,(0,255,255),1)
    if confirm1 > 0:
        cv2.line(img, pts1[0], pts1[1], (0, 255, 0), 1)
        cv2.line(img, pts1[1], pts1[3], (0, 255, 0), 1)
        cv2.line(img, pts1[2], pts1[3], (0, 255, 0), 1)
        cv2.line(img, pts1[2], pts1[0], (0, 255, 0), 1)
    if confirm2 > 0:
        cv2.line(img, pts2[0], pts2[1], (0, 255, 0), 1)
        cv2.line(img, pts2[1], pts2[3], (0, 255, 0), 1)
        cv2.line(img, pts2[2], pts2[3], (0, 255, 0), 1)
        cv2.line(img, pts2[2], pts2[0], (0, 255, 0), 1)
    if confirm1 > 0 and confirm2 > 0:
        cv2.line(img, pts1[0], pts2[0], (0, 255, 0), 1)
        cv2.line(img, pts1[1], pts2[1], (0, 255, 0), 1)
        cv2.line(img, pts1[2], pts2[2], (0, 255, 0), 1)
        cv2.line(img, pts1[3], pts2[3], (0, 255, 0), 1)
    # ptsBottom = np.array([[points[0+8],points[1+8]],[points[2+8],points[3+8]],[points[4+8],points[5+8]],[points[6+8],points[7+8]]], np.int32)
    # cv2.polylines(img,[ptsBottom],True,(255,0,0),1)
    return 0


def draw_3DBBox(img, points):
    pts = np.array([[points[0], points[1]], [points[2], points[3]], [points[4], points[5]], [points[6], points[7]]],
                   np.int32)
    cv2.polylines(img, [pts], True, (255, 0, 0), 1)
    ptsBottom = np.array(
        [[points[0 + 8], points[1 + 8]], [points[2 + 8], points[3 + 8]], [points[4 + 8], points[5 + 8]],
         [points[6 + 8], points[7 + 8]]], np.int32)
    cv2.polylines(img, [ptsBottom], True, (255, 0, 0), 1)
    index = 0
    while index <= 3:
        i = index * 2
        cv2.line(img, (points[i], points[i + 1]), (points[i + 8], points[i + 9]), (255, 0, 0), 1)
        index = index + 1
    return img


def draw_singlelane(img, line_points, color, linevisibility):
    length_line = len(line_points)
    for i in (range(length_line - 1)):
        cv2.line(img, line_points[i], line_points[i + 1], (255, 255, 255), 1)
        if linevisibility[i] == 1:
            cv2.circle(img, line_points[i], 2, color, -1)
        else:
            cv2.circle(img, line_points[i], 2, (50, 50, 50), -1)
    return img


def process_single(img, bboxFileJson):
    bboxes = {}
    culledBoxes = {}
    lenth1 = len(bboxFileJson["bboxes"])
    [height, width, ch] = img.shape
    for i in range(lenth1):
        if bboxFileJson["bboxes"][i].get('truncation') is not None:
            bboxes[tuple(bboxFileJson["bboxes"][i]["bbox"])] = bboxFileJson["bboxes"][i]["truncation"]
    lenth2 = len(bboxFileJson["bboxesCulled"])
    # #print(lenth2)
    for j in range(lenth2):
        if bboxFileJson["bboxesCulled"][j].get('truncation') is not None:
            culledBoxes[tuple(bboxFileJson["bboxesCulled"][j]["bbox"])] = bboxFileJson["bboxesCulled"][j]["truncation"]
    img_bbox = draw_bboxes(img, bboxes, culledBoxes)
    return img_bbox


def draw3dbox(img, points):
    lineColor = (255, 0, 0)
    thickness = 1
    points = np.array(points, int)

    for i in [0, 3]:
        cv2.line(img, points[i], points[1], (0, 0, 255), thickness, lineType=cv2.LINE_AA)
        cv2.line(img, points[i], points[2], (0, 0, 255), thickness, lineType=cv2.LINE_AA)
    for i in [4, 7]:
        cv2.line(img, points[i], points[5], (0, 255, 0), thickness, lineType=cv2.LINE_AA)
        cv2.line(img, points[i], points[6], (0, 255, 0), thickness, lineType=cv2.LINE_AA)
    for i in range(4):
        cv2.line(img, points[i], points[i + 4], (255, 255, 255), thickness, lineType=cv2.LINE_AA)


def fisheye_project_points(blank_image, object_points, rvec, tvec, K, D, image_shape):
    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)

    # 使用鱼眼相机模型进行投影
    image_points, _ = cv2.fisheye.projectPoints(object_points, rvec, tvec, K, D)

    image_points = image_points.reshape([8, 2])

    # 创建一个空白图像

    # 定义点的颜色（以BGR格式）
    color = (0, 255, 0)
    # 绘制点
    # 在图像上绘制投影后的点
    thickness = 2
    i = 0
    lineColor = (255, 0, 0)
    points = np.array(image_points, int)

    for i in [0, 3]:
        cv2.line(blank_image, points[i], points[1], (0, 0, 255), thickness, lineType=cv2.LINE_AA)
        cv2.line(blank_image, points[i], points[2], (0, 0, 255), thickness, lineType=cv2.LINE_AA)
    for i in [4, 7]:
        cv2.line(blank_image, points[i], points[5], (0, 255, 0), thickness, lineType=cv2.LINE_AA)
        cv2.line(blank_image, points[i], points[6], (0, 255, 0), thickness, lineType=cv2.LINE_AA)
    for i in range(4):
        cv2.line(blank_image, points[i], points[i + 4], (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    cv2.circle(blank_image, points[0], 1, (255, 0, 0), -1)
    cv2.circle(blank_image, points[1], 1, (255, 128, 128), -1)
    cv2.circle(blank_image, points[2], 1, (0, 128, 128), -1)

    return blank_image


def process_single3D(img, bboxFileJson, settingFileJson):
    boxes3DCarHead = []
    boxes3DCarTail = []
    [height, width, ch] = img.shape

    fx = settingFileJson["fx"]
    fy = settingFileJson["fy"]
    cx = settingFileJson["cx"]
    cy = settingFileJson["cy"]

    camera_matrix = np.mat([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], np.float32)

    distortion = np.array(settingFileJson["distortion"], np.float32)
    distortionfisheye = np.array(settingFileJson["distortionfisheye"], np.float64)
    T0 = np.array([0, 0, 0], np.float32)
    R0 = np.array([0, 0, 0], np.float32)

    if settingFileJson["distortionenable"] or not settingFileJson["fisheyeenable"]:
        for bbox3D in bboxFileJson["bboxes3D"]:
            cube = np.array([
                bbox3D["relativeCornerFrontLeftTop"],
                bbox3D["relativeCornerFrontRightTop"],
                bbox3D["relativeCornerFrontLeftBottom"],
                bbox3D["relativeCornerFrontRightBottom"],
                bbox3D["relativeCornerRearLeftTop"],
                bbox3D["relativeCornerRearRightTop"],
                bbox3D["relativeCornerRearLeftBottom"],
                bbox3D["relativeCornerRearRightBottom"]
            ])
            tmp = cube.copy()
            cube[:, 2] = np.maximum(tmp[:, 2], 0)
            result, _ = cv2.projectPoints(cube, R0, T0, camera_matrix, distortion)
            result = result.reshape([8, 2])
            draw3dbox(img, result)
        return img
    elif settingFileJson["fisheyeenable"]:
        # 图像形状
        image_shape = (HEIGHT, WIDTH, 3)
        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)  # 旋转向量
        tvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)  # 平移向量

        for bbox3D in bboxFileJson["bboxes3D"]:
            cube = np.array([[
                bbox3D["relativeCornerFrontLeftTop"],
                bbox3D["relativeCornerFrontRightTop"],
                bbox3D["relativeCornerFrontLeftBottom"],
                bbox3D["relativeCornerFrontRightBottom"],
                bbox3D["relativeCornerRearLeftTop"],
                bbox3D["relativeCornerRearRightTop"],
                bbox3D["relativeCornerRearLeftBottom"],
                bbox3D["relativeCornerRearRightBottom"]
            ]])
            tmp = cube.copy()
            cube[0, :, 2] = np.maximum(tmp[0, :, 2], 0)
            camera_matrix = np.mat([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            projected_image = fisheye_project_points(img, cube, rvec, tvec, camera_matrix, distortionfisheye,
                                                     image_shape)
        return projected_image

    else:
        lenth3 = len(bboxFileJson["bboxes3D"])
        for k in range(lenth3):
            boxes3DCarHead = []
            boxes3DCarTail = []
            cube = np.array([
                bboxFileJson["bboxes3D"][k]["relativeCornerFrontLeftTop"],
                bboxFileJson["bboxes3D"][k]["relativeCornerFrontRightTop"],
                bboxFileJson["bboxes3D"][k]["relativeCornerFrontLeftBottom"],
                bboxFileJson["bboxes3D"][k]["relativeCornerFrontRightBottom"],
                bboxFileJson["bboxes3D"][k]["relativeCornerRearLeftTop"],
                bboxFileJson["bboxes3D"][k]["relativeCornerRearRightTop"],
                bboxFileJson["bboxes3D"][k]["relativeCornerRearLeftBottom"],
                bboxFileJson["bboxes3D"][k]["relativeCornerRearRightBottom"]
            ])
            tmp = cube.copy()
            cube[:, 2] = np.maximum(tmp[:, 2], 0)
            boxes3DCarHead.append([cube[0][0] / cube[0][2] * fx + cx, cube[0][1] / cube[0][2] * fy + cy])
            boxes3DCarHead.append([cube[1][0] / cube[1][2] * fx + cx, cube[1][1] / cube[1][2] * fy + cy])
            boxes3DCarHead.append([cube[2][0] / cube[2][2] * fx + cx, cube[2][1] / cube[2][2] * fy + cy])
            boxes3DCarHead.append([cube[3][0] / cube[3][2] * fx + cx, cube[3][1] / cube[3][2] * fy + cy])
            boxes3DCarTail.append([cube[4][0] / cube[4][2] * fx + cx, cube[4][1] / cube[4][2] * fy + cy])
            boxes3DCarTail.append([cube[5][0] / cube[5][2] * fx + cx, cube[5][1] / cube[5][2] * fy + cy])
            boxes3DCarTail.append([cube[6][0] / cube[6][2] * fx + cx, cube[6][1] / cube[6][2] * fy + cy])
            boxes3DCarTail.append([cube[7][0] / cube[7][2] * fx + cx, cube[7][1] / cube[7][2] * fy + cy])
            draw_3DB(img, boxes3DCarHead, boxes3DCarTail)
        return img


def process_line(img, laneFileJson):
    lenth1ane = len(laneFileJson["lane_lines"])
    for i in range(lenth1ane):
        laneline = laneFileJson["lane_lines"][i]
        if "uv" in laneline:
            line_2d = laneline["uv"]
            if laneline["attribute"] == 0:
                color = (255, 0, 0)
            elif laneline["attribute"] == 1:
                color = (0, 255, 0)
            elif laneline["attribute"] == 2:
                color = (0, 0, 255)
            elif laneline["attribute"] == 3:
                color = (0, 255, 255)
            elif laneline["attribute"] == 4:
                color = (255, 255, 0)
            else:
                color = (255, 0, 255)
            # print("attribute is " + laneline["attribute"])
            if len(line_2d[0]) != len(line_2d[1]):
                print("---------data loss!!!-------------")
                break
            else:
                line_points = []
                lane_range = len(line_2d[0])
                for j in range(lane_range):
                    line_points.append((int(line_2d[0][j]), int(line_2d[1][j])))
                img = draw_singlelane(img, line_points, color, laneline["visibility"])
    if "laneMark_StopLines" in laneFileJson:
        lengthstopline = len(laneFileJson["laneMark_StopLines"])
        for j in range(lengthstopline):
            stopline = laneFileJson["laneMark_StopLines"][j]
            if "stopline_projection" in stopline:
                stopline_points = []
                for nums in range(len(stopline["stopline_projection"])):
                    project_points = stopline["stopline_projection"][nums]
                    stopline_points.append((int(project_points["u"]), int(project_points["v"])))
                    cv2.circle(img, (int(project_points["u"]), int(project_points["v"])), 5, (150, 150, 0), -1)

    if "laneMark_CrossWalks" in laneFileJson:
        lengthcrosswalk = len(laneFileJson["laneMark_CrossWalks"])
        for k in range(lengthcrosswalk):
            crosswalk = laneFileJson["laneMark_CrossWalks"][k]
            if "crosswalk_projection" in crosswalk:
                crosswalk_points = []
                for numc in range(len(crosswalk["crosswalk_projection"])):
                    project_pointc = crosswalk["crosswalk_projection"][numc]
                    crosswalk_points.append((int(project_pointc["u"]), int(project_pointc["v"])))
                    cv2.circle(img, (int(project_pointc["u"]), int(project_pointc["v"])), 5, (0, 150, 150), -1)

    if "laneMark_TrafficRoadMark" in laneFileJson:
        lengthTrafficRoadMark = len(laneFileJson["laneMark_TrafficRoadMark"])
        for k in range(lengthTrafficRoadMark):
            TrafficRoadMark = laneFileJson["laneMark_TrafficRoadMark"][k]
            if "TrafficRoadMark_projection" in TrafficRoadMark:
                TrafficRoadMark_points = []
                confirm = 1
                OutPoints = TrafficRoadMark["TrafficRoadMark_projection"]
                for numM in range(len(OutPoints)):
                    project_pointM = OutPoints[numM]
                    if ((int(project_pointM["u"]) > 0) and (int(project_pointM["u"]) < WIDTH) and (
                            int(project_pointM["v"]) > 0) and (int(project_pointM["v"]) < HEIGHT)):
                        TrafficRoadMark_points.append((int(project_pointM["u"]), int(project_pointM["v"])))
                        cv2.circle(img, (int(project_pointM["u"]), int(project_pointM["v"])), 5, (150, 20, 150), -1)
                    else:
                        confirm = 0
                if confirm > 0:
                    tmp = TrafficRoadMark_points[len(TrafficRoadMark_points) - 1]
                    TrafficRoadMark_points.append(tmp)
                    for numss in range(len(TrafficRoadMark_points) - 1):
                        cv2.line(img, TrafficRoadMark_points[numss], TrafficRoadMark_points[numss + 1], (255, 255, 255),
                                 1)
                        # cv2.line(img,TrafficRoadMark_points[3],TrafficRoadMark_points[4],(255,255,255),1)
    if "laneMark_ParkingSpaces" in laneFileJson:
        lengthParkingSpace = len(laneFileJson["laneMark_ParkingSpaces"])
        # print("range of lengthParkingSpace")
        for P in range(lengthParkingSpace):
            parkingspace = laneFileJson["laneMark_ParkingSpaces"][P]
            if "Parking_InnerCorner_projection" in parkingspace:
                InnerPoints = parkingspace["Parking_InnerCorner_projection"]
                confirm = 1
                tiltBboxes = []
                pts_parking = []
                for nums in range(len(InnerPoints)):
                    project_points = InnerPoints[nums]
                    if ((int(project_points["u"]) > 0) and (int(project_points["u"]) < WIDTH) and (
                            int(project_points["v"]) > 0) and (int(project_points["v"]) < HEIGHT)):
                        pts_parking.append((int(project_points["u"]), int(project_points["v"])))
                    else:
                        confirm = 0
                pts_parking.append(InnerPoints[0])
                if confirm > 0:
                    for numss in range(len(InnerPoints)):
                        # cv2.line(img,pts_parking[numss],pts_parking[numss+1],(255,255,0),1)
                        cv2.circle(img, pts_parking[numss], 2, (0, 0, 255), -1)
            if "Parking_OutCorner_projection" in parkingspace:
                OutPoints = parkingspace["Parking_OutCorner_projection"]
                parking_points = []
                confirm = 1
                pts_parking = []
                for nums in range(len(OutPoints)):
                    project_points = OutPoints[nums]
                    if ((int(project_points["u"]) > 0) and (int(project_points["u"]) < WIDTH) and (
                            int(project_points["v"]) > 0) and (int(project_points["v"]) < HEIGHT)):
                        pts_parking.append((int(project_points["u"]), int(project_points["v"])))
                    else:
                        confirm = 0
                pts_parking.append(OutPoints[0])
                if confirm > 0:
                    for numss in range(len(OutPoints) - 1):
                        cv2.line(img, pts_parking[numss], pts_parking[numss + 1], (255, 255, 0), 1)
                    # cv2.circle(img, pts_parking[4], 2, (255, 255, 255), -1)
                    # cv2.circle(img, pts_parking[1], 2, (0, 0, 10), -1)
                    # cv2.circle(img, pts_parking[2], 2, (100, 100, 100), -1)
                    # cv2.circle(img, pts_parking[3], 2, (0, 255, 120), -1)
            if "Parking_PaintLeft_projection" in parkingspace:
                LeftPaintPoints = parkingspace["Parking_PaintLeft_projection"]
                confirm = 1
                pts_parking = []
                for nums in range(len(LeftPaintPoints)):
                    project_points = LeftPaintPoints[nums]
                    if ((int(project_points["u"]) > 0) and (int(project_points["u"]) < WIDTH) and (
                            int(project_points["v"]) > 0) and (int(project_points["v"]) < HEIGHT)):
                        pts_parking.append((int(project_points["u"]), int(project_points["v"])))
                    else:
                        confirm = 0
                pts_parking.append(LeftPaintPoints[0])
                if confirm > 0:
                    for numss in range(len(LeftPaintPoints) - 1):
                        cv2.line(img, pts_parking[numss], pts_parking[numss + 1], (120, 255, 0), 1)
            if "Parking_PaintRight_projection" in parkingspace:
                RightPaintPoints = parkingspace["Parking_PaintRight_projection"]
                confirm = 1
                pts_parking = []
                for nums in range(len(RightPaintPoints)):
                    project_points = RightPaintPoints[nums]
                    if ((int(project_points["u"]) > 0) and (int(project_points["u"]) < WIDTH) and (
                            int(project_points["v"]) > 0) and (int(project_points["v"]) < HEIGHT)):
                        pts_parking.append((int(project_points["u"]), int(project_points["v"])))
                    else:
                        confirm = 0
                pts_parking.append(RightPaintPoints[0])
                if confirm > 0:
                    for numss in range(len(RightPaintPoints) - 1):
                        cv2.line(img, pts_parking[numss], pts_parking[numss + 1], (255, 120, 0), 1)

    return img


def process_batch(img_folder, img_folder1, path):
    img_path = img_folder + '.jpg'
    # print(img_path)
    cameraInfoPath = os.path.join(img_folder1, 'CameraInfo.json')
    bbox_file = open(cameraInfoPath, "rb")
    bboxFileJson = json.load(bbox_file)
    if "bboxes" in bboxFileJson:
        img = cv2.imread(img_path)
        img_bbox = process_single(img, bboxFileJson)
        if os.path.exists(path):
            os.remove(path)
        cv2.imwrite(path + '.jpg', img_bbox)


def process_batch3D(img_folder, img_folder1, path, path_setting):
    img_path = img_folder + '.jpg'
    # print(img_path)
    cameraInfoPath = os.path.join(img_folder1, 'CameraInfo.json')
    dumpSettingPath = os.path.join(path_setting, 'DumpSettings.json')
    bbox_file = open(cameraInfoPath, "rb")
    setting_file = open(dumpSettingPath, "rb")
    bboxFileJson = json.load(bbox_file)
    settingFileJson = json.load(setting_file)
    settingFileJson = settingFileJson["camera"]
    if "bboxes3D" in bboxFileJson:
        img = cv2.imread(img_path)
        img_bbox = process_single3D(img, bboxFileJson, settingFileJson)
        if os.path.exists(path):
            os.remove(path)
        cv2.imwrite(path + '.jpg', img_bbox)


def process_Lane(img_folder, img_folder1, path):
    img_path = img_folder + '.jpg'
    # print(img_path)
    LaneInfoPath = os.path.join(img_folder1, 'LaneInfo.json')
    if not os.path.exists(LaneInfoPath):
        return
    lane_file = open(LaneInfoPath, "rb")
    laneFileJson = json.load(lane_file)
    if "lane_lines" in laneFileJson:
        img = cv2.imread(img_path)
        img_lane = process_line(img, laneFileJson)
        if os.path.exists(path):
            os.remove(path)
        cv2.imwrite(path + '.jpg', img_lane)


def SaveBytearrayTojpg(BinaryData, DstfilePath, width, height):
    k = np.zeros((height, width, 3), np.uint8)
    num = 0
    k = np.frombuffer(BinaryData, dtype=np.uint8)
    k = k.reshape(height, width, 3)

    cv2.imwrite(DstfilePath + ".jpg", k)


def SaveColorTojpg(SrcfilePath, DstfilePath):
    f = open(SrcfilePath, 'rb')
    filedata = f.read()
    filesize = f.tell()
    f.close()
    BinaryData = bytearray(filedata)
    SaveBytearrayTojpg(BinaryData, DstfilePath, WIDTH, HEIGHT)


def SaveDepthTojpg(SrcfilePath, DstfilePath):
    floatData = np.fromfile(SrcfilePath, np.float32)
    k = np.frombuffer(floatData, dtype=np.float32)
    k = k.reshape(HEIGHT, WIDTH)
    cv2.imwrite(DstfilePath + ".jpg", k)


SegmentationColorTable = np.array([
    [0, 0, 0],
    [107, 142, 35],
    [70, 70, 70],
    [128, 64, 128],
    [220, 20, 60],
    [153, 153, 153],
    [0, 0, 142],
    [0, 0, 0],
    [119, 11, 32],
    [190, 153, 153],
    [70, 130, 180],
    [244, 35, 232],
    [240, 240, 240],
    [220, 220, 0],
    [102, 102, 156],
    [250, 170, 30],
    [152, 251, 152],
    [255, 0, 0],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [111, 74, 0],
    [180, 165, 180],
    [81, 0, 81],
    [150, 100, 100],
    [220, 220, 0],
    [169, 11, 32],
    [250, 170, 160],
    [230, 150, 140],
    [150, 120, 90],
    [151, 124, 0],
    [70, 120, 120],
    [70, 12, 120],
    [70, 120, 12],
    [0, 120, 120],
    [200, 120, 120],
    [70, 200, 120],
    [70, 120, 200],
    [100, 0, 0],
    [250, 120, 120],
    [70, 0, 250],
    [140, 100, 100],
    [160, 160, 160],
    [170, 10, 10],
    [130, 100, 10],
    [170, 100, 10],
    [170, 10, 100],
    [170, 170, 170],
    [100, 20, 10],
])


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='BGR'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.int64)  # uint8
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb[labelmap == label] = SegmentationColorTable[label]

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def SegmentationTypeToColor(type):
    if type >= 0 and type < 32:
        return SegmentationColorTable[type]
    return SegmentationColorTable[0]


def SaveRleSegmentationTojpg(SrcfilePath, DstfilePath, DstfilePath1):
    f = open(SrcfilePath, 'rb')
    rledata = f.read()
    rle_header_ = RLE_Header()
    try:
        rlx_pix_data = decode_rlesegmentation(rledata, rle_header_)
    except:
        rlx_pix_data = rledata

    pred_color = colorEncode(rlx_pix_data, SegmentationColorTable)
    cv2.imwrite(DstfilePath + ".jpg", rlx_pix_data)
    cv2.imwrite(DstfilePath1 + ".jpg", pred_color)


def SaveCompressedSegmentationTojpg(SrcfilePath, DstfilePath, DstfilePath1):
    f = open(SrcfilePath, 'rb')
    zlibdata = f.read()
    f.close()
    try:
        filedata = zlib.decompress(zlibdata)
    except:
        filedata = zlibdata
    BinaryData = bytearray(filedata)

    k = np.frombuffer(BinaryData, dtype=np.uint8)
    k = k.reshape(HEIGHT, WIDTH)
    pred_color = colorEncode(k, SegmentationColorTable)

    cv2.imwrite(DstfilePath + ".jpg", k)
    cv2.imwrite(DstfilePath1 + ".jpg", pred_color)


def SaveCompressedInstanceTojpg(SrcfilePath, DstfilePath):
    f = open(SrcfilePath, 'rb')
    zlibdata = f.read()
    f.close()
    try:
        filedata = zlib.decompress(zlibdata)
    except:
        filedata = zlibdata

    BinaryData = bytearray(filedata)
    SaveBytearrayTojpg(BinaryData, DstfilePath, WIDTH, HEIGHT)


FlowFlagBitTable = [
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0]
]


def SaveFlowFlagBitTojpg(SrcfilePath, DstfilePath):
    f = open(SrcfilePath, 'rb')
    filedata = f.read()
    f.close()
    BinaryData = bytearray(filedata)
    k = np.frombuffer(BinaryData, dtype=np.uint8)
    k = k.reshape(HEIGHT, WIDTH)
    pred_color = colorEncode(k, FlowFlagBitTable)

    cv2.imwrite(DstfilePath + ".jpg", pred_color)


def DestPath(inPath):
    dstPath = inPath
    if os.path.exists(dstPath):
        os.remove(dstPath)
    return dstPath

def convertDump(fi, fi_d, files1, output_path, filepath, case):
    for fi_z in files1:
        fi_k = os.path.join(fi_d, fi_z)
        if not os.path.isfile(fi_k):
            return

        if fi_z == "Segmentation" and SEGMENTATION:
            Segmentation_path = output_path + '/Segmentation/'
            color_Segmentation_path = output_path + '/SegmentationView/'
            if not os.path.exists(Segmentation_path):
                os.mkdir(Segmentation_path)

            if not os.path.exists(color_Segmentation_path):
                os.mkdir(color_Segmentation_path)

            SaveRleSegmentationTojpg(fi_k, DestPath(Segmentation_path + 'Segmentation' + fi), DestPath(
                color_Segmentation_path + 'Segmentation_View' + fi))

        elif fi_z == "Color" and COLOR:
            case_root = 'simone_datasets'
            lidar_file_path = os.path.join(case_root, case, "lidar_top")
            formatted_date = utils.get_pcd_cretatime(lidar_file_path)
            # filename = case+"-"+formatted_date+"-0800"+"__LIDAR_TOP__"+save_lidar_path
            Color_path = output_path
            Color_path1 = output_path + '/Colorbox/'
            Color_path2 = output_path + '/Color3Dbox/'
            Color_path3 = output_path + '/ColorOpenLane/'
            if not os.path.exists(Color_path):
                os.mkdir(Color_path)
            parts = output_path.split(os.sep)
            filename = case + "-" + formatted_date + "-0800" + "__" + parts[-1]+"__" + str(int(timestart+int(fi)*1/60*1000*1000))
            SaveColorTojpg(fi_k, DestPath(os.path.join(Color_path, filename)))
            'scene-2004-2024-08-14-03-19-46-0800__CAM_FRONT__1533151649512000'
            # print(DestPath(Color_path + 'Color' + fi))
            if DrawBB:
                if not os.path.exists(Color_path1):
                    os.mkdir(Color_path1)
                process_batch(DestPath(Color_path + 'Color' + fi), fi_d, DestPath(Color_path1 + 'Colorbox' + fi))
            if Draw3DBB:
                if not os.path.exists(Color_path2):
                    os.mkdir(Color_path2)
                process_batch3D(DestPath(Color_path + 'Color' + fi), fi_d, DestPath(Color_path2 + 'Color3Dbox' + fi),
                                filepath)
            if DrawOpenLane:
                if not os.path.exists(Color_path3):
                    os.mkdir(Color_path3)
                process_Lane(DestPath(Color_path + 'Color' + fi), fi_d, DestPath(Color_path3 + 'ColorOpenLane' + fi))
        elif fi_z == "WorldNormals" and NORMALS:
            Normal_path = args.output + '/Normal/'
            if not os.path.exists(Normal_path):
                os.mkdir(Normal_path)
            SaveColorTojpg(fi_k, DestPath(Normal_path + 'Normal' + fi))
            print(DestPath(Normal_path + 'Normal' + fi))
        elif fi_z == "DepthPlanner" and DEPTH:
            DepthPlanner_path = output_path + '/DepthPlanner/'
            if not os.path.exists(DepthPlanner_path):
                os.mkdir(DepthPlanner_path)
            SaveDepthTojpg(fi_k, DestPath(DepthPlanner_path + 'DepthPlanner' + fi))

        elif fi_z == "Instance" and INSTANCE:
            Instance_path = output_path + '/Instance/'
            if not os.path.exists(Instance_path):
                os.mkdir(Instance_path)
            SaveCompressedInstanceTojpg(fi_k, DestPath(Instance_path + 'Instance' + fi))

        if fi_z == "BaseColor" and BaseColor:
            BaseColor_path = output_path + '/BaseColor/'
            if not os.path.exists(BaseColor_path):
                os.mkdir(BaseColor_path)
            SaveColorTojpg(fi_k, DestPath(BaseColor_path + 'BaseColor' + fi))
        if fi_z == "BaseColorForward" and BaseColor:
            BaseColor_path = output_path + '/BaseColorForward/'
            if not os.path.exists(BaseColor_path):
                os.mkdir(BaseColor_path)
            SaveColorTojpg(fi_k, DestPath(BaseColor_path + 'BaseColorForward' + fi))
        if fi_z == "VisualFlow" and VisualFlow:
            VisualBackFlow_path = output_path + '/VisualFlow/'
            if not os.path.exists(VisualBackFlow_path):
                os.mkdir(VisualBackFlow_path)
            SaveColorTojpg(fi_k, DestPath(VisualBackFlow_path + 'VisualFlow' + fi))
        if fi_z == "VisualFlowForward" and VisualFlow:
            VisualFlowForward_path = output_path + '/VisualFlowForward/'
            if not os.path.exists(VisualFlowForward_path):
                os.mkdir(VisualFlowForward_path)
            SaveColorTojpg(fi_k, DestPath(VisualFlowForward_path + 'VisualFlowForward' + fi))
        if fi_z == "FlowFlagBit" and VisualFlow:
            FlowFlagBit_path = output_path + '/FlowFlagBit/'
            if not os.path.exists(FlowFlagBit_path):
                os.mkdir(FlowFlagBit_path)
            SaveFlowFlagBitTojpg(fi_k, DestPath(FlowFlagBit_path + 'FlowFlagBit' + fi))
        if fi_z == "FlowFlagBitForward" and VisualFlow:
            FlowFlagBitForward_path = output_path + '/FlowFlagBitForward/'
            if not os.path.exists(FlowFlagBitForward_path):
                os.mkdir(FlowFlagBitForward_path)
            SaveFlowFlagBitTojpg(fi_k, DestPath(FlowFlagBitForward_path + 'FlowFlagBitForward' + fi))
    print("Finish : ", fi_d)


# noinspection PyInterpreter
def gci(filepath, case, args):
    files = os.listdir(filepath)

    files = sorted(files,  key=extract_number)

    complete = False
    counter = 0  # 添加计数器
    print(counter)

    with ThreadPoolExecutor(max_workers=THREADCOUNT) as t:
        threadList = []
        for fi in files:
            if fi == 'DumpSettings.json':
                continue
            fi_d = os.path.join(filepath, fi)
            if not os.path.isdir(fi_d):
                continue

            # 根据计数器调整输出路径
            if counter % 6 == 0:
                args.output = os.path.join('nuscenes_v1.0', 'samples', dest)
            else:
                args.output = os.path.join('nuscenes_v1.0', 'sweeps', dest)

            # 处理文件的代码
            files1 = os.listdir(fi_d)
            dumpSettingPath = os.path.join(filepath, 'DumpSettings.json')
            setting_file = open(dumpSettingPath, "rb")
            settingFileJson = json.load(setting_file)
            settingFileJson = settingFileJson["camera"]
            global WIDTH
            WIDTH = settingFileJson["width"]
            global HEIGHT
            HEIGHT = settingFileJson["height"]

            r = t.submit(convertDump, fi, fi_d, files1, args.output, filepath, case)
            threadList.append(r)

            counter += 1  # 计数器增加

        try:
            while not complete:
                complete = all(task.done() for task in threadList)
                time.sleep(3)
        except KeyboardInterrupt:
            print("Keyboard Interrupt Stop...")
            for task in threadList:
                task.cancel()
            t.shutdown(wait=True)

def copy_lidar_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file_name in os.listdir(src_dir):
        if file_name.endswith('.pcd'):
            full_file_name = os.path.join(src_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy2(full_file_name, dest_dir)


camera_mapping_dict = {'front': 'CAM_FRONT', 'front_left': 'CAM_FRONT_LEFT', 'front_right': "CAM_FRONT_RIGHT",
                       'rear': 'CAM_BACK', 'rear_left': 'CAM_BACK_LEFT', 'rear_right': 'CAM_BACK_RIGHT'}

lidar_mapping_dict = {'lidar_top': 'LIDAR_TOP'}
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Model related arguments
    print('Usage: RAW2jpg.py --input=inputpath --output=outpath')
    parser.add_argument('--input', default='front', required=False,
                        help="a path input file")
    parser.add_argument('--output', required=False,
                        help="a path out file")

    args = parser.parse_args()

    # Raw2JPG for camera and output results to dest path
    case_root = 'simone_datasets'
    cases = os.listdir(case_root)
    # for case in cases:
    for index, case in enumerate(cases):
        timestart = utils.timestart + index * 5 * 60 * 1000000
        camera_file_path = os.path.join(case_root, case)
        for src, dest in camera_mapping_dict.items():
            camera_file_path_ = os.path.join(camera_file_path, src)
            CurworkingDir = os.path.realpath(camera_file_path_)
            args.output = os.path.join('nuscenes_v1.0', 'samples', dest)
            if len(SUBFOLDER) > 0:
                CurworkingDir = os.path.join(CurworkingDir, SUBFOLDER)

            if not os.path.exists(args.output):
                os.mkdir(args.output)

            print('start converting........')
            start_time = time.time()
            gci(CurworkingDir, case, args)
            print('over.........', time.time() - start_time)
