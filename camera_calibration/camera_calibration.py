import numpy as np
import cv2 as cv
import glob
import os

class CameraCalibration:
    def __init__(self, chessboard_size, frame_size):
        """
        Initialize the CameraCalibration object.
        """
        self.chessboard_size = chessboard_size
        self.frame_size = frame_size
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objpoints = []
        self.imgpoints = []

    def find_chessboard_corners(self, image_path):
        """
        This function populates objpoints and imgpoints if chessboard corners are found in the image.
        """
        img = cv.imread(image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, self.chessboard_size, None)

        if ret:
            self.objpoints.append(self.objp)
            corners2 = cv.cornerSubPix(gray, corners, self.chessboard_size, (-1, -1), self.criteria)
            self.imgpoints.append(corners)

            cv.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
            cv.imshow('img', cv.resize(img, (640, 480)))
            cv.waitKey(2000)

    def calibrate_camera(self,calibration_file):
        """
        This function uses objpoints and imgpoints to calibrate the camera and saves the calibration results.
        """
        # ret, self.camera_matrix, self.distortion_coefficients, self.rvecs, self.tvecs = cv.calibrateCamera(
        # self.objpoints, self.imgpoints, self.frame_size[::-1], None, None)
        ret, self.camera_matrix, self.distortion_coefficients, self.rvecs, self.tvecs = cv.calibrateCamera(
        self.objpoints, self.imgpoints, self.frame_size[::-1], None, None)   

        rotMat, _ = cv.Rodrigues(self.rvecs[0])

        calibration_path = os.path.join('src', calibration_file)
        np.savez(calibration_path, 
                mtx=np.asarray(self.camera_matrix, dtype=np.longdouble), 
                rotMat=np.asarray(rotMat, dtype=np.longdouble), 
                tvec=np.asarray(self.tvecs[0], dtype=np.longdouble),
                dist=np.asarray(self.distortion_coefficients, dtype=np.longdouble))

    def undistort_images(self, input_folder='raw_images', output_folder='calibrated_images'):
        """
        This function undistorts images using the calibration results and saves them to the specified output folder.
        """
        images_path = os.path.join(input_folder, '*.bmp')
        images = glob.glob(images_path)
        os.makedirs(output_folder, exist_ok=True)

        for image_path in images:
            img = cv.imread(image_path)
            h, w = img.shape[:2]
            new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
                self.camera_matrix, self.distortion_coefficients, (w, h), 1, (w, h))

            mapx, mapy = cv.initUndistortRectifyMap(
                self.camera_matrix, self.distortion_coefficients, None, new_camera_matrix, (w, h), 5)
            dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]

            output_path = os.path.join(output_folder, os.path.basename(image_path))
            cv.imwrite(output_path, dst)

    def calculate_reprojection_error(self):
        """
         This function calculates the reprojection error using the calibrated camera parameters and prints the result.
        """
        mean_error = 0

        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i], self.camera_matrix, self.distortion_coefficients
            )
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            mean_error += error

        print("Total error: {}".format(mean_error / len(self.objpoints)))

# Example usage
chessboard_size = (14, 12)
frame_size = (2448, 2080)

calibration = CameraCalibration(chessboard_size, frame_size)
calibration_images_path = os.path.join(os.path.dirname(__file__), 'calibration_image.bmp')
calibration_images = glob.glob(calibration_images_path)

for image_path in calibration_images:
    calibration.find_chessboard_corners(image_path)

calibration.calibrate_camera(calibration_file='calibration_data.npz')

calibration.undistort_images()
calibration.calculate_reprojection_error()

cv.destroyAllWindows()
