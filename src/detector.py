"""
LEGO Part Detection Module
@author:  
"""

import cv2
import json
import copy
from os.path import join, dirname
from .utils import calculate_intersection_over_union, ROI
from ultralytics import YOLO

# Global variables for visualization
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.8
angle_color = (0, 255, 0)
det_color = (0, 0, 255)
other_color = (255, 0, 0)
thickness = 1
X = 0
Y = 1

# Path for JSON configuration file
FILE_PATH = dirname(__file__)
JSON_CONFIG = join(FILE_PATH, "config.json")

class Detection:
    """
    Represents a detected LEGO part.

    Attributes:
        x, y, w, h: Coordinates and dimensions of the bounding box around the detected part.
        label: Label of the detected part.
        confidence: Confidence score of the detection.
        angle: Angle of the part, if applicable.
        best_iou: Best intersection over union (IoU) score of the part.
    """
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.label = "label_u"
        self.confidence = 0.0
        self.angle = 0
        self.best_iou = 0

    def from_yolo_output(self, result, names):
        """Initialize Detection object from YOLO output."""
        self.x = int(round(result.xyxy[0][0]))
        self.y = int(round(result.xyxy[0][1]))
        self.w = int(round(result.xyxy[0][2]) - self.x)
        self.h = int(round(result.xyxy[0][3]) - self.y)
        self.label = names[int(result.cls[0])]
        self.confidence = round(result.conf[0], 2)

    def from_bbox(self, bb, label, confidence):
        """Initialize Detection object from bounding box."""
        self.x = bb[0]
        self.y = bb[1]
        self.w = bb[2] - bb[0]
        self.h = bb[3] - bb[1]
        self.label = label
        self.confidence = confidence

    def get_bb(self):
        """Get bounding box coordinates."""
        return [self.x, self.y, self.x+self.w, self.y+self.h]

    def get_centre(self):
        """Get center coordinates of the bounding box."""
        return [self.x+self.w/2, self.y+self.h/2]

    def visualize(self, image, color):
        """Visualize the detection on the image."""
        cv2.rectangle(image, (self.x, self.y), 
                      (self.x+self.w, self.y+self.h), color, 1)
        text = f"label:{self.label} angle:{self.angle} "
        cv2.putText(image, text, (self.x, self.y),
                    font, fontScale, color, thickness, cv2.LINE_AA)

class DetectorOutput:
    """
    Represents the output of the LEGO part detection.

    Attributes:
        label: Label of the detected part.
        pick_point: Coordinates of the pick-up point.
        angle: Angle of the detected part.
        parts_count: Total count of LEGO parts in the image.
        unknown: Flag indicating if the detected part is unknown.
        side: Flag indicating if side picking is needed.
        do_shake: Flag indicating if shaking the parts is needed.
        frame: Processed frame with visualizations.
    """
    def __init__(self):
        self.label = None
        self.pick_point = (0.0, 0.0)
        self.angle = 0
        self.parts_count = 0
        self.unknown = True
        self.frame = None
        self.side = False
        self.do_shake = False

    def from_list(self, parameters):
        """Initialize DetectorOutput from a list of parameters."""
        self.label = parameters[0]
        self.pick_point = parameters[1]
        self.angle = parameters[2]
        self.parts_count = parameters[3]
        self.unknown = parameters[4]
        self.side = parameters[5]
        self.do_shake = parameters[6]
        self.frame = parameters[7]

    def from_values(self, label, pick_point, angle, parts_count, unknown, side, do_shake, frame):
        """Initialize DetectorOutput from individual values."""
        self.label = label
        self.pick_point = pick_point
        self.angle = angle
        self.parts_count = parts_count
        self.unknown = unknown
        self.side = side
        self.do_shake = do_shake
        self.frame = frame

class Detector:
    """
    LEGO part detection class.

    Attributes:
        size: Size of the processed image.
        roi: Region of interest for detection.
        angle_image_roi: ROI for angle estimation.
        x_factor: Factor for x-coordinate conversion.
        y_factor: Factor for y-coordinate conversion.
        x_offset: Offset for x-coordinate.
        y_offset: Offset for y-coordinate.
        last_centre: Coordinates of the last detected part's center.
        same_place_count: Counter for consecutive detections at the same place.
        camera_calibrated: Flag indicating if the camera is calibrated.
    """
    def __init__(self, size=640, roi=[100, 100, 740, 740], angle_image_roi=[20, 45, 623, 565]):
        self.size = size
        self.roi = ROI(roi)
        self.angle_roi = ROI(angle_image_roi)
        self.x_factor = (self.roi.x2 - self.roi.x1) / size
        self.y_factor = (self.roi.y2 - self.roi.y1) / size
        self.x_offset = self.roi.x1
        self.y_offset = self.roi.y1
        self.last_centre = [0, 0]
        self.same_place_count = 0
        self.camera_calibrated = False

    def load_model(self, model_path, confidence, iou_threshold):
        """
        Load the YOLO model and parameters from JSON config file.

        Args:
            model_path: Path to the YOLO model.
            confidence: Confidence threshold for detection.
            iou_threshold: Intersection over Union threshold for detection.
        """
        self.model = YOLO(model_path) 
        self.model.conf = confidence  
        self.model.iou = iou_threshold

        # Load parameters from JSON config file
        with open(JSON_CONFIG) as f:
            json_dict = json.load(f)

        self.offsets = json_dict["offset"]
        self.side_pick = json_dict["side_pick"]
        self.throw = json_dict["throw"]
        self.distance = json_dict["distance"]
        self.up_edge_thresh  = json_dict["up_edge_thresh"]
        self.down_edge_thresh = json_dict["down_edge_thresh"]
        self.left_edge_thresh = json_dict["left_edge_thresh"]
        self.right_edge_thresh = json_dict["right_edge_thresh"]
        f.close()

    def detect_image(self, frame, visualize=False):
        """Detect LEGO parts in the given image."""
        if frame is None or len(frame) == 0:
            print("No image")
            return DetectorOutput()

        # Preprocess image
        frame = cv2.resize(frame, (self.size, self.size))

        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame_gray = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self.camera_calibrated:
            _, self.bw_background = cv2.threshold(frame_gray, 250, 255, cv2.THRESH_BINARY)
            self.invert_bw = cv2.bitwise_not(self.bw_background)
            self.bw_background = self.bw_background / 255
            self.bw_background = self.bw_background.astype('uint8')
            self.camera_calibrated = True

        _, bw = cv2.threshold(frame_gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Perform YOLO inference
        results = self.model([frame])

        detections = []
        names = results[0].names
        best_score = 0.0
        index = 0

        for res in results:
            for box in res.boxes:
                det = Detection()
                det.from_yolo_output(box.numpy(), names)
                if det.confidence > best_score:
                    best_score = det.confidence
                detections.append(det)

        # Estimate angles
        contours, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        angles = []
        for i, c in enumerate(contours):
            if hier[0][i][3] == -1:
                continue

            area = cv2.contourArea(c)

            if area < 200 or area > 100000:
                continue
            
            x, y, w, h = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)

            angle = int(rect[2]) + 3
            min_rect_w = rect[1][0]
            min_rect_h = rect[1][1]

            angle_detection = Detection()
            angle_detection.from_bbox([int(x), int(y), 
                                       int(x + w), int(y + h)],
                                       "angle", 0.0)

            if min_rect_w < min_rect_h:
                angle = 90 - angle
            else:
                angle = -angle

            for detection in detections:
                iou = calculate_intersection_over_union(angle_detection.get_bb(), detection.get_bb())
                if iou > detection.best_iou:
                    detection.angle = angle
                    detection.best_iou = iou

            angle_detection.angle = angle
            angles.append(angle_detection)

        unknown = False

        temp_angles = []
        for detection in detections:
            temp_angles.append(detection.angle)

        angles_without_dets = []
        for detection in angles:
            if detection.angle in temp_angles:
                temp_angles.pop(temp_angles.index(detection.angle))
            else:
                angles_without_dets.append(detection)

        if len(angles) > len(detections):
            object_count = len(angles)
        else:
            object_count = len(detections)

        if len(detections) == 0:
            unknown = True
            if len(angles) == 0:
                detections.append(Detection())
            else:
                angles[0].label = "unknown"
                detections.append(angles[0])

        if detections[index].label[-1] == "u" and detections[index].label.split("_")[0] in self.throw:
            unknown = True

        label = detections[index].label.split("_")[0]
        pixel_centre = detections[index].get_centre()
        pick_point = [round(pixel_centre[X] * self.x_factor + self.x_offset, 2), 
                      round(pixel_centre[Y] * self.y_factor + self.y_offset, 2)]
        pick_angle = detections[index].angle

        do_shake = False
        for i in range(len(detections)):
            if i == index:
                continue
            else:
                centre = detections[i].get_centre()
                euclidean_distance = ((pixel_centre[X] - centre[X])**2 +
                                      (pixel_centre[Y] - centre[Y])**2) ** 0.5
                if euclidean_distance <= self.distance:
                    do_shake = True
                    break
        if not do_shake:
            for det in angles_without_dets:
                centre = det.get_centre()
                euclidean_distance = ((pixel_centre[X] - centre[X])**2 +
                                      (pixel_centre[Y] - centre[Y])**2) ** 0.5
                if euclidean_distance <= self.distance:
                    do_shake = True
                    break

        side = False
        if label in self.side_pick:
            side = True

        if label in self.offsets:
            pick_angle += self.offsets[label]

        if pixel_centre[Y] <= self.up_edge_thresh or pixel_centre[Y] >= self.down_edge_thresh:
            unknown = True
        if pixel_centre[X] <= self.left_edge_thresh or pixel_centre[X] >= self.right_edge_thresh:
            unknown = True

        euclidean_distance = ((pixel_centre[X] - self.last_centre[X])**2 +
                              (pixel_centre[Y] - self.last_centre[Y])**2) ** 0.5
        if euclidean_distance < 10:
            self.same_place_count += 1
            if self.same_place_count >= 20:
                self.same_place_count = 0
            pick_angle += 10 * self.same_place_count
        else:
            self.same_place_count += 0
            self.last_centre = copy.copy(pixel_centre)

        if visualize:
            for i in range(len(detections)):
                if i == index:
                    detections[i].visualize(frame, det_color)
                else:
                    detections[i].visualize(frame, other_color)           
            for angle_det in angles_without_dets:
                angle_det.visualize(frame, angle_color)

        output = DetectorOutput()
        output.from_values(label, pick_point, pick_angle, object_count, unknown, side, do_shake, frame)
        return output
