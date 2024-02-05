import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from multiprocessing import get_context, TimeoutError
import numpy as np
import time
import os
import json
from camera import Camera


class faceDetection():
    """
    Callable class that detects faces in images.
    In this case the point between the eyes is used as the point location of the face.

    This can be done because cvzone detects FaceMeshes including eyes.
    """
    def __init__(self):
        self.detector = FaceMeshDetector(maxFaces=1, staticMode=True)

    def __call__(self, img):
        img, faces = self.detector.findFaceMesh(img, draw=False)
        # face landmarks 145 and 374 should be the right and the left eye. The position between these points is
        # calculated directly
        faces = [(np.array(face[145]) + np.array(face[374]))/2 for face in faces]
        return faces

    def calibrate_cameras(self, rows, columns, n_images=5, scaling=1):
        """
        This method calibrates the cameras of the faceDetection one by one.
        This is it calculates the internal camera matrix based on a list of images shot by the camera of a chess board.
        Soo keep an image of a chessboard ready!

        :param rows:        Number of rows in the chessboard/checkerboard
        :param columns:     Number of column in the checkerboard
        :param n_images:    Number of images that should be taken for calibration.
        :param scaling:     Size of a checkerboard square in real world (this will calibrate the cameras coordinate
                            system to real world units of your liking)
        :return:
        """


class faceHandeler():
    """
    Callable class, that handles which face from a list of face landmarks should be focussed on.
    This could be important if multiple faces are detected.

    Currently, this is just a placeholder, and it will just return the first face in the list
    """

    def __init__(self):
        pass

    def __call__(self, faces):
        """

        :param faces: List of lists of face landmarks
        :return:      list of face landmarks for a single face
        """

        if len(faces) == 0:
            return faces
        faces = faces[0]
        return faces


class faceTriangulation():
    def stero_calibrate(self, rows=5, columns=7, n_images=6, scaling=0.01):
        """
        Calibrates the cameras for stereovision via several chess-board images.
        The calibration function should be automated in the future by automatic chess-board detection

        """
        # open cv can only detect inner corners, so reducing the rows and columns by one
        rows -= 1
        columns -= 1

        # change this if stereo calibration not good.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        # coordinates of squares in the checkerboard world space
        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp = scaling * objp

        # frame dimensions. Frames should be the same size.
        width = self.cameras[0].read()[1].shape[1]
        height = self.cameras[0].read()[1].shape[0]

        # Pixel coordinates in image space of checkerboard
        imgpoints_1 = []
        imgpoints_2 = []

        # World coordinates of the checkerboard. (World coordinate system has its origin in the bottom left corner of
        # the checkerboard.
        objpoints = []

        while len(imgpoints_1) < n_images:
            # reading the frames form the cameras
            success1, img1 = self.cameras[0].read()
            success2, img2 = self.cameras[1].read()

            if success1 and success2:
                # Displaying the images
                cv2.imshow(f"{self.cameras[0]}", img1)
                cv2.imshow(f"{self.cameras[1]}", img2)
                key = cv2.waitKey(1)
                if key == 27:  # esc stops the calibration process
                    print("Aborting stereo calibration.")
                    return
                # press any key except esc to take the frame for calibration
                if key > 0:
                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)

                    # Detecting the checkerboard corners
                    corner_success1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
                    corner_success2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)

                    if corner_success2 and corner_success1:

                        # Refining the detection
                        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                        # Showing the detection
                        cv2.drawChessboardCorners(img1, (rows, columns), corners1, corner_success1)
                        cv2.drawChessboardCorners(img2, (rows, columns), corners2, corner_success2)
                        cv2.imshow(f'Detection {self.cameras[0]}', img1)
                        cv2.imshow(f'Detection {self.cameras[1]}', img2)
                        key = cv2.waitKey(0)
                        # press any key to continue
                        if key > 0:
                            # storing the realworld/ image-space coordinate pairs
                            objpoints.append(objp)
                            imgpoints_1.append(corners1)
                            imgpoints_2.append(corners2)
                            cv2.destroyWindow(f'Detection {self.cameras[0]}')
                            cv2.destroyWindow(f'Detection {self.cameras[1]}')
        # prerform stereo calibration on accumulated objectpoints
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                      imgpoints_1,
                                                                      imgpoints_1,
                                                                      self.cameras[0].camera_matrix,
                                                                      self.cameras[0].distortion,
                                                                      self.cameras[1].camera_matrix,
                                                                      self.cameras[1].distortion,
                                                                      (width, height),
                                                                      criteria=criteria,
                                                                      flags=stereocalibration_flags)

        # Matrix that rotates the coordinate system of the second camera to match the first.
        self.rotation_matrix = R
        # Matrix that translates the coordinate system of the second camera to match the first.
        self.translation_matrix = T

        print(f'Stereo-calibration error: {ret}')
        print(f'Translation Matrix: {T}')
        print(f'Rotation Matrix: {R}')

        self.set_config("stereo_calibration_error", ret)
        self.set_config("translation_matrix", T)
        self.set_config("rotation_matrix", R)
        return


    def set_config(self, config_name, value):
        try:
            with open('face_position_configurations.json', 'r') as json_file:
                configurations = json.load(json_file)
        except FileNotFoundError:
            print("No camera configuration file found!")
            return
        # converting np arrays to lists

        if isinstance(value, (np.ndarray, list, tuple)):

            value = np.array(value).tolist()


        # storing the stream of the camera in a JSON format.
        configurations[self.name][config_name] = value

        # saving the updated dict as JSON
        with open('face_position_configurations.json', 'w') as json_file:
            json.dump(configurations, json_file)

    def get_config(self, config_name, value):
        try:
            with open('face_position_configurations.json', 'r') as json_file:
                configurations = json.load(json_file)
        except FileNotFoundError:
            print("No camera configurations file found!")
            return

        # storing the stream of the camera in a JSON format.
        value = configurations[self.name][config_name]

        return value

    def triangulate_face(self, img1, img2):
        """
        :param img1 & img2:  Rectified images from both cameras
        :return:
        """



class spatialFacePosition(faceTriangulation):
    """
    # Notes:
    - Camera Synchronization?

    This is the main class used for calculating the spatial face position.
    It inherits from faceTriangulation and uses its parent class to calculate the vector pointing from the
    "main"-camera to the face(or position between both eyes). This vector is based in an 3D-coordinate system defined
    by the camera view direction and rotation. Using this vector and the normal vector of the screen plane, a
    screen-rotation vector and the vector connecting the camera and the screen center, the vector connecting the screen
    center and the face will be calculated. This vector should be based on a coordinate system starting with the screen
    center, the screen normal and the screen rotation vector.

    This information will later be used in the Unity-based animation to change the view frustum to match
    the view as if the monitor would be a window.

    The triangulation has to be fast and robust to mimic the view out of a window effectively.

    #Proposed Class Structure:

    Initialization:
    -During initialization the vector connecting the screen center and the main-camera, the screen-normal and a vector
     Orthogonal to the screen-normal, which is used to set the rotation of the screen (a parallel vector to the lower
     edge of the screen.
        -Maybe in the future an automatic process for determining these attributes can be developed.
            (like: index finger to middle of screen and screen corners, then calculation)

    Methods:
    - triangulate_head_position:
        This method uses the calculated vector pointing from the camera to the face to calculate the spatial position
        of the head in relation to the monitor.


    """

    def __init__(self,
                 name='',
                 face_detection=faceDetection(),
                 handle_faces=faceHandeler()
                 ):
        """
        Aside from attributing the monitor center vector and the monitor normal vector to the class instance, also
        the

        :param face_detection:        [callable Object] Function or callable class that takes in an image and returns
                                      the coordinates of the face or position between the eyes
        :param handle_faces:          Callable object that chooses the right face from the detected list of faces.
        :param name:                  Name of this configuration. If a setup with the same name was already initialized
                                      before, all parameters from the jason file from the previous initialization
                                      will be used.


        """
        super().__init__()
        # Loading the configurations for this setup!
        self.name = name
        self.stereo_calibrated = False    # will be set to True during stereo-calibration or load config
        if not self.load_config():
            print(f"No SpatialFacePosition configuration named {self.name} found!")
        self.face_detection = face_detection

        self.handle_faces = handle_faces

        # Initializing the cameras
        self.cameras = [Camera(camera) for camera in self.cameras]

    def __str__(self):
        return self.name

    def load_config(self):
        """
        If a configuration with the same name was already initialized, there will be a JSON file with
        configurations and calibrations for this setup, it will be loaded by this function.
        """
        try:
            with open('face_position_configurations.json', 'r') as json_file:
                configurations = json.load(json_file)
                self.cameras = configurations[self.name]["cameras"]
                self.monitor_center = configurations[self.name]["monitor_center"]
                self.monitor_normal = configurations[self.name]["monitor_normal"]
                self.monitor_rotation = configurations[self.name]["monitor_rotation"]
                # loading parameters that are only stored after stereo-calibration
                try:
                    self.rotation_matrix = configurations[self.name]["rotation_matrix"]
                    self.translation_matrix = configurations[self.name]["translation_matrix"]
                    self.stereo_calibrated = True
                except KeyError:
                    self.stereo_calibrated = False
            return True

        except FileNotFoundError:
            print("No configurations.json file found!")
            return False

    def __call__(self):
        """
        Reads both streams, loads just one frame and predicts the spatial position of the head in relation to the
        screen.

        :return:    Success face_pos img1 (with face annotations) img2 (with face annotations)
        """
        #####!!!! rewrite for an abitrary ammount of cameras!
        sucess1, img1 = self.cameras[0].read()
        sucess2, img2 = self.cameras[0].read()

        if not sucess2 or not sucess1:
            print('Could not read both Cams')
            return False

        ##### Frame Rectification #####
        # Undisort and rectify images using the stereoMap generated during calibration

        img1 = cv2.remap(img1, self.stereoMap1_x, self.stereoMap1_y, cv2.INTER_LANCZOS4)
        img2 = cv2.remap(img2, self.stereoMap2_x, self.stereoMap2_y, cv2.INTER_LANCZOS4)
        face1, face2 = self.triangulate_face(img1, img2)

        return self.current_depth, face1, face2, img1, img2

