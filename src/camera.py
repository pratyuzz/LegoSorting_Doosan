"""
Camera Class
@author: 
"""
import numpy as np
import cv2

class Camera:
    def __init__(self, calib_data_path, roi=[400, 400, 800, 800], init_time=100000):
        """
        Initialize the Camera object.

        Parameters:
            calib_data_path (str): Path to the calibration data numpy file.
            roi (list): List containing [x1, y1, x2, y2] coordinates of the region of interest (default is [400, 400, 800, 800]).
            init_time (int): Initial time value (default is 100000).
        """
        calib_data = np.load(calib_data_path) 
        self.mtx = calib_data["mtx"]  # Camera matrix
        self.dist = calib_data["dist"]  # Distortion coefficients
        self.roi = roi  
        self.init_time = init_time 
        self.cam = None

    def initialize_camera(self):        
        """
        Initialize the camera capture.

        """
        # Initialize the camera itself
        self.cam = cv2.VideoCapture(0) # Indicates the index of the camera device to be used
        # self.cam.set(cv2.CAP_PROP_EXPOSURE,-12) 
        if not self.cam.isOpened():
            raise ValueError("Failed to open the camera.")

    # def get_image(self):
    #     ret, frame = self.cam.read()
    #     if not ret:
    #         raise ValueError("Failed to capture frame from the camera.")
        
    #     # Print the raw frame (before undistortion)
    #     print("Raw Frame:", frame)


    def get_image(self):
        """
        Capture and undistort an image from the camera.

        Returns:
            Undistorted image within the region of interest.

        """
        ret, frame = self.cam.read()
        if not ret:
            raise ValueError("Failed to capture frame from the camera.")        

        # Undistort the captured image using the calibration data and extract region of interest
        image = cv2.undistort(frame, self.mtx, self.dist, None)
        return image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]

    def __del__(self):
        """
        Release the camera capture when the Camera object is deleted.
        """
        if self.cam is not None:
            self.cam.release()
