"""
LEGO Sorting Utilities
@author: 
"""

import numpy as np
from numpy.linalg import inv

class ROI():
    def __init__(self, roi):
        """
        Initialize RegionOfInterest object with given coordinates.

        Parameters:
            roi (tuple): Tuple containing (x1, y1, x2, y2) coordinates of the region.
        """
        self.x1 = roi[0]
        self.y1 = roi[1]
        self.x2 = roi[2]
        self.y2 = roi[3]

def calculate_intersection_over_union(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Parameters:
        box1 (tuple): Tuple containing (x1, y1, x2, y2) coordinates of the first bounding box.
        box2 (tuple): Tuple containing (x1, y1, x2, y2) coordinates of the second bounding box.

    Returns:
        float: Intersection over Union (IoU) value.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    intersection_area = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if intersection_area == 0:
        return 0

    box1_area = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    box2_area = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def check_overlapping_bounding_boxes(bb1, bb2):
    """
    Check if two bounding boxes overlap.

    Parameters:
        bb1 (tuple): Tuple containing (x1, y1, x2, y2) coordinates of the first bounding box.
        bb2 (tuple): Tuple containing (x1, y1, x2, y2) coordinates of the second bounding box.

    Returns:
        bool: True if bounding boxes overlap, False otherwise.
    """
    return not ((bb1[0] >= bb2[2]) or (bb1[2] <= bb2[0]) or (bb1[3] <= bb2[1]) or (bb1[1] >= bb2[3]))

def ground_project_point(image_point, camera_matrix, rotation_matrix, translation_vector, z=0.0):
    """
    Project image point onto ground plane.

    Parameters:
        image_point (tuple): Tuple containing (x, y) coordinates of the image point.
        camera_matrix (numpy.ndarray): Camera matrix.
        rotation_matrix (numpy.ndarray): Rotation matrix.
        translation_vector (numpy.ndarray): Translation vector.
        z (float): Desired z-coordinate on the ground plane (default is 0.0).

    Returns:
        numpy.ndarray: World coordinates (x, y, z) of the projected point.
    """
    cam_matrix = np.asarray(camera_matrix)
    inv_rotation = inv(rotation_matrix)
    inv_cam = inv(cam_matrix)

    uv_point = np.ones((3, 1))
    uv_point[0, 0] = image_point[0]
    uv_point[1, 0] = image_point[1]

    temp_mat = np.matmul(np.matmul(inv_rotation, inv_cam), uv_point)
    temp_mat2 = np.matmul(inv_rotation, translation_vector)

    scale = (z + temp_mat2[2, 0]) / temp_mat[2, 0]
    world_coordinates = np.matmul(inv_rotation, (np.matmul(scale * inv_cam, uv_point) - translation_vector))

    # Assert that the projected z-coordinate is approximately equal to the desired z
    assert int(abs(world_coordinates[2] - z) * (10 ** 8)) == 0
    world_coordinates[2] = z

    return world_coordinates

