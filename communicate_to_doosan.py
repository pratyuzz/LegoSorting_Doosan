"""
LEGO Sorting Robot Server
This script sets up a server to communicate with a LEGO sorting robot. It receives requests from the robot, processes them, 
and generates responses containing commands for the robot.

@author: 
"""

import sys
import socket
import select
import cv2
import os
from os.path import join, dirname
from src.camera import Camera
from src.calibrator import Calibrator
from src.detector import Detector

# # Suppressing warnings
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

CHECKPOINT_NAME = "yolov8_legoset.pt"
HOST = ''
PORT = 50000
PROJECT_PATH = dirname(__file__)
YOLOV8_FOLDER = join(PROJECT_PATH, "models", "yolov8")
MODEL_PATH = join(PROJECT_PATH, "models", "checkpoints", CHECKPOINT_NAME)
CALIBRATION_DATA_PATH = join(PROJECT_PATH, "src", "calibration_data.npz")

class RobotCommand:
    """
    Represents a command to be sent to the robot.

    Attributes:
        mode (int): Operating mode of the robot.
        part_count (int): Count of LEGO parts detected.
        label (str): Label of the detected LEGO part.
        side_pick (int): Flag indicating if the part is to be side-picked.
        x (float): X-coordinate of the pick-up point.
        y (float): Y-coordinate of the pick-up point.
        angle (int): Angle of the detected LEGO part.
    """
    def __init__(self) -> None:
        self.mode = 0
        self.part_count = 0
        self.label = 0
        self.side_pick = 0
        self.x = 0
        self.y = 0
        self.angle = 0

    def __str__(self) -> str:
        """
        String representation of the RobotCommand.

        Returns:
            str: Formatted string representation of the command.
        """
        return "{:d};{:d};{};{:d};{:.2f};{:.2f};{:d}\r\n" \
            .format(self.mode, self.part_count, self.label, self.side_pick, self.x, self.y, self.angle)
    

def GenerateResponse(detector : Detector, calibrator: Calibrator, camera:Camera) -> str:
    """
    Generate response command based on detected LEGO parts.

    Args:
        detector (Detector): Object for detecting LEGO parts in the image.
        calibrator (Calibrator): Object for calibrating camera.
        camera (Camera): Object representing the camera.

    Returns:
        str: Command to be sent to the robot.
    """
    
    command = RobotCommand()
    frame = camera.get_image()
    detection = detector.detect_image(frame, True)

    if detection.label is None:
        print("Unable to get a frame from the camera")
        return None

    command.x, command.y = calibrator.project_point(detection.pick_point)
    command.part_count = detection.parts_count
    command.label = detection.label
    command.side_pick = detection.side
    command.angle = detection.angle

    if detection.parts_count == 0:
        command.mode = 9
    elif detection.unknown:
        command.mode = 2
    elif detection.do_shake:
        command.mode = 5
    else:
        command.mode = 1

    cv2.imshow("frame", detection.frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(1)

    return str(command)

if __name__ == "__main__":
    # Initialize camera, detector, and calibrator
    roi = [270, 0, 2190, 1920]
    camera = Camera(CALIBRATION_DATA_PATH,roi, init_time=125000)
    camera.initialize_camera()
    detector = Detector(640,roi)
    detector.load_model(MODEL_PATH, 0.90, 0.45)
    calibrator = Calibrator(CALIBRATION_DATA_PATH)
    
    # while True:
    #     response = GenerateResponse(detector, calibrator, camera)
    #     print(response)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        print("#########################")
        print("Starting server")
        print("#########################")

        s.bind((HOST, PORT))
        s.settimeout(1)
        s.listen()
        try:
        
            while True:

                try:
                    conn, addr = s.accept()
                except(KeyboardInterrupt):
                    print("keyboard interrupt")
                    sys.exit(1)
                except(socket.timeout):
                    continue

                with conn:
                    s.setblocking(0)
                    print('Connected by', addr)
                    try:
                        while True:
                            ready = select.select([conn], [], [], 5)[0]
                            if ready:
                                data = conn.recv(32)
                                if not data:
                                    continue
                            else:
                                continue

                            msg = data.decode("utf-8")

                            print("received,", msg[0])
                            if msg[0] == "T":

                                response = None
                                tries = 0
                                while response is None and tries < 5:
                                    response = GenerateResponse(detector, calibrator, camera)
                                    tries += 1

                                if response is None:
                                    print("Detection failed")
                                    continue

                                data = bytes(response, 'utf-8')
                                print("Sending:", data)
                                
                                conn.sendall(data)
                                
                                
                    except (KeyboardInterrupt, ConnectionAbortedError, ConnectionResetError):
                        print(f"Got Error or Interrupt, closing the connection")
                        print("Waiting for new connection, close with ctrl+C")
                        pass
                    s.setblocking(1)
                    s.settimeout(3)
        except KeyboardInterrupt as interrupt:
            print("keyboard interrupt, exiting....", interrupt)
            s.close()
            sys.exit(1)


