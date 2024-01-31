"""
Calibrator Class
@author:
"""

from .utils import ground_project_point
import numpy as np

class Calibrator():
    def __init__(self, calib_data_path):
        """
        Initialize the CalibrationAssistant object.

        Parameters:
            calib_data_path (str): Path to the calibration data numpy file.
        """
        calib_data = np.load(calib_data_path)
        self.camera_matrix  = calib_data["mtx"]  # Camera matrix
        self.rotation_matrix  = calib_data["rotMat"]  # Rotation matrix
        self.translation_vector  = calib_data["tvec"]  # Translation vector

        # Define offsets for calibration adjustments
        self.x_offset = -8  # Offset from the lower edge to the backlight origin
        self.y_offset = -10  # Offset from the left edge to the backlight origin

        # Size of each chessboard tile used in calibration
        self.tile_size = 10

    def project_point(self, centre_point):
        """
        Project pixel coordinates to millimeter coordinates according to the camera calibration data.

        Parameters:
            centre_point (tuple): Tuple containing (x, y) pixel coordinates of the centre point.

        Returns:
            tuple: Tuple containing (x_mm, y_mm) millimeter coordinates.
        """
        # Project pixel coordinates to millimeter coordinates using calibration data
        projected_point = ground_project_point(centre_point, self.camera_matrix, self.rotation_matrix, self.translation_vector)
        
        # Adjust projected point to millimeter coordinates with offsets and tile size
        x_mm = round((projected_point[0][0] + self.x_offset) * -1 * self.tile_size, 2)
        y_mm = round((projected_point[1][0] + self.y_offset) * self.tile_size, 2)
        
        return x_mm, y_mm
