import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from multiprocessing import get_context, TimeoutError
import numpy as np
import time
import os
import json




class Camera():
    def __init__(self,
                 name="Camera"
                 ):
        """
        :param name:   Name of the camera-setup stored in camera_configuration.json
        """
        self.name = name
        # Initially calibrated will be False by default, but in the load_config method, calibrated will be set to True
        # if a camera matrix, and distortion coefficients are found.
        self.calibrated = False
        # same for fisheye
        self.fisheye = False
        # loading the camera-configurations form camera_configurations.json
        if not self.load_configs():
            print(f"No camera-configuration for name: {name} found!")
        if self.resolution:
            self.stream = cv2.VideoCapture(self.stream, cv2.CAP_DSHOW)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[1])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[0])
        else:
            self.stream = cv2.VideoCapture(self.stream)



    def read(self):
        return self.stream.read()

    def __str__(self):
        return self.name

    def undistort_image(self, img):
        """
        Undistorts an image using the camera_matrix and the distortion values obtained by camera calibration.
        :param img:             Image to undistort
        :param camera_matrix:   Camera matrix obtained by camera claibration
        :param distortion:      Distortion parameters obtained by camera calibration.
        :param optimized
        _camera_matrix:         Camera matrix optimized by cv2.getOptoimalNewCameraMatrix

        :return:                Undistorted image, and new camera_matrix
        """
        if self.fisheye:
            img = self.undistort_fisheye(img)
            return img

        img = cv2.undistort(img, self.camera_matrix, self.distortion, None, self.optimized_camera_matrix)
        return img

    def undistort_fisheye(self, img):
        h, w = img.shape[:2]
        dim = img.shape[:2][::-1]

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.fisheye_camera_matrix,
                                                         self.fisheye_distortion, np.eye(3),
                                                         self.fisheye_optimized_camera_matrix,
                                                         [w, h],
                                                         cv2.CV_16SC2)
        img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return img

    def set_config(self, config_name, value):
        try:
            with open('camera_configuration.json', 'r') as json_file:
                configurations = json.load(json_file)
        except FileNotFoundError:
            print("No camera configuration file found!")
            return

        # converting np arrays to lists
        if isinstance(value, (np.ndarray, list, tuple)):
            value = np.array(value).tolist()


        # storing the config value of the camera in a JSON format.
        configurations[self.name][config_name] = value

        # saving the updated dict as JSON
        with open('camera_configuration.json', 'w') as json_file:
            json.dump(configurations, json_file)

    def get_config(self, config_name, value):
        try:
            with open('camera_configuration.json', 'r') as json_file:
                configurations = json.load(json_file)
        except FileNotFoundError:
            print("No camera configurations file found!")
            return

        # storing the stream of the camera in a JSON format.
        value = configurations[self.name][config_name]

        return value

    def load_configs(self):
        """
        If a camera with the same name was already configured, there will probably be a JSON file with
        configurations and calibrations for this setup, it will be loaded by this function.
        """
        try:
            with open('camera_configuration.json', 'r') as json_file:
                configurations = json.load(json_file)
                self.stream = configurations[self.name]["stream"]
                try:
                    self.resolution  = configurations[self.name]["resolution"]
                except KeyError: # Going with standard resolution 480 * 640.
                    self.resolution = False

                # trying to load attributes that are only added to the camera during calibration.
                try:
                    self.projection_error = configurations[self.name]["projection_error"]
                    self.camera_matrix = np.array(configurations[self.name]["camera_matrix"])
                    self.distortion = np.array(configurations[self.name]["distortion"])
                    self.optimized_camera_matrix = np.array(configurations[self.name]["optimized_camera_matrix"])

                    self.current_camera_matrix = self.optimized_camera_matrix  # gets overwritten if fisheye
                    self.calibrated = True
                except KeyError as E:
                    print('Could not load all values', E)
                    self.calibrated = False
                # trying to load fisheye calibration
                # fisheye calibration is optional and will only throw an error if fisheye is set to true in config.
                # and not all params could be loaded
                try:
                    self.fisheye = configurations[self.name]["fisheye"]

                    if self.fisheye:
                        self.fisheye_camera_matrix = np.array(configurations[self.name]["fisheye_camera_matrix"])
                        self.fisheye_distortion = np.array(configurations[self.name]["fisheye_distortion"])
                        self.fisheye_optimized_camera_matrix = np.array(
                            configurations[self.name]["fisheye_optimized_camera_matrix"])

                        self.current_camera_matrix = self.fisheye_optimized_camera_matrix
                                                                                # if the camera is fisheye calibrated
                                                                                # the current camera matrix is
                                                                                # overwritten by the
                        #                                                         fisheye_camera_matrix
                except KeyError as E:
                    if self.fisheye:
                     print("Error while loading fisheye calibration, maybe camera was not fisheye calibrated jet", E)

            return True

        except FileNotFoundError:
            print("No camera_configuration.json file found!")
            return False


    def calibrate(self, rows=7, columns=9, n_images=10, scaling=0.025, fisheye=False):
        """
        Thanks to: https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html

        # Calibration for normal cameras (pin hole)

        Calculates the camera matrix for a camera based on a checkerboard-pattern.
        :param rows:        Number of rows in chessboard-pattern.
        :param columns:     Number of columns in chessboard-pattern.
        :param scaling:     Realworld size of a chess-board square to scale the coordinate system.
                            I will try to keep all units in meters so a good initial value for this will be 0.01 or 1 cm

        :n_images:          Number of photos that will be taken for calibration.
        :param fisheye:     Indicates if fisheye calibration should be used
        :return:
        """
        assert n_images > 0, "n_images must be larger than zero!"
        # performing fisheye_calibration first and on top of that the normal calibration.
        if fisheye:
            try:
                # only performing fisheye calibration if it was not already fisheye calibrated before
                _ = self.fisheye_optimized_camera_matrix
            except Exception:
                self.fisheye_calibrate(rows=rows, columns=columns, n_images=n_images, scaling=scaling)

        # Only chessboard corners with all four sides being squares can be detected. (B W) Therefor the detectable
        # chessboard is one smaller in number of rows and columns.                   (W B)
        rows -= 1
        columns -= 1
        # termination criteria
        # If no chessboard-pattern is detected, change this... Don't ask me what to change about it!
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, lower left corner of chessboard will be world coordinate (0, 0, 0)
        objp = np.zeros((columns * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp = scaling * objp

        # Chessboard pixel coordinates (2D)
        imgpoints = []
        # Chessboard pixel coordinates in world-space (3D). Coordinate system defined by chessboard.
        objpoints = []

        while True:
            success, img = self.read()
            img_old = np.array(img)
            # If fisheye calibration was performed normal calibration is performed on top of the undistortion of the
            # fisheye calibration
            if fisheye:
                img = self.undistort_fisheye(img)
                img_old = np.array(img)
            if len(imgpoints) >= n_images:  # Processed enough pictures
                break
            if success:
                cv2.imshow(f'Camera: {self}', img)
                k = cv2.waitKey(1)
                if k > 0:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                    # localizing the chessboard corners in s       image-space.
                    ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
                    if ret:
                        # trying to improve the detection of the chessboard corners!
                        corners = cv2.cornerSubPix(gray,
                                                   corners,
                                                   (11, 11),  # size of kernel!
                                                   (-1, -1),
                                                   criteria)
                        cv2.drawChessboardCorners(img, (rows, columns), corners, ret)
                        for i, [corner] in enumerate(corners): # Check if enumeration is consistent
                            cv2.putText(img, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 0, 255), 1)
                        cv2.imshow(f'Chessboard corners; Camera: {self}', img)
                        key = cv2.waitKey(0)
                        if key & 0xFF == ord('s'):  # press s to switch the ordering of the corners
                            cv2.destroyWindow(f'Chessboard corners; Camera: {self}')
                            corners = corners[::-1]
                            # drawing the new corners
                            cv2.drawChessboardCorners(img_old, (rows, columns), corners, ret)
                            for i, [corner] in enumerate(corners):  # Check if enumeration is consistent
                                cv2.putText(img_old, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                                            (0, 0, 255), 1)
                            cv2.imshow(f'Chessboard corners; Camera: {self}', img_old)
                            cv2.waitKey(0)

                        cv2.destroyWindow(f'Chessboard corners; Camera: {self}')

                        objpoints.append(objp)
                        imgpoints.append(corners)

        # Camera resolution
        success, img = self.read()
        height = img.shape[0]
        width = img.shape[1]




        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        # saving the optimized camera matrix
        height, width = img.shape[:2]
        optimized_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1,
                                                                   (width, height))


        print('rmse:', ret)
        print('camera matrix:\n', mtx)
        print('optimized camera matrix:\n', optimized_camera_matrix)
        print('distortion coeffs:\n', dist)


        self.projection_error = ret
        self.camera_matrix = mtx
        self.optimized_camera_matrix = optimized_camera_matrix
        self.distortion = dist


        # Updating the camera_configuration.json file
        self.set_config("projection_error", self.projection_error)
        self.set_config("camera_matrix", self.camera_matrix)
        self.set_config("distortion", self.distortion)
        self.set_config("optimized_camera_matrix", self.optimized_camera_matrix)


        # closing the window after calibration
        cv2.destroyWindow(f'Camera: {self}')
        self.calibrated = True
        cv2.destroyAllWindows()
        return

    def fisheye_calibrate(self, rows=7, columns=9, n_images=10, scaling=0.025):
        """
        Thanks to: https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html

        # Calibration for fisheye cameras

        Calculates the camera matrix for a camera based on a checkerboard-pattern.
        :param rows:        Number of rows in chessboard-pattern.
        :param columns:     Number of columns in chessboard-pattern.
        :param scaling:     Realworld size of a chess-board square to scale the coordinate system.
                            I will try to keep all units in meters so a good initial value for this will be 0.01 or 1 cm

        :n_images:          Number of photos that will be taken for calibration.

        :return:
        """
        assert n_images > 0, "n_images must be larger than zero!"

        # Only chessboard corners with all four sides being squares can be detected. (B W) Therefor the detectable
        # chessboard is one smaller in number of rows and columns.                   (W B)
        rows -= 1
        columns -= 1
        # termination criteria
        # If no chessboard-pattern is detected, change this... Don't ask me what to change about it!
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, lower left corner of chessboard will be world coordinate (0, 0, 0)
        objp = np.zeros((columns * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp = scaling * objp

        # Chessboard pixel coordinates (2D)
        imgpoints = []
        # Chessboard pixel coordinates in world-space (3D). Coordinate system defined by chessboard.
        objpoints = []

        while True:
            success, img = self.read()
            img_old = np.array(img)
            if len(imgpoints) >= n_images:  # Processed enough pictures
                break
            if success:
                cv2.imshow(f'Camera: {self}', img)
                k = cv2.waitKey(1)
                if k > 0:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                    # localizing the chessboard corners in image-space.
                    ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
                    if ret:
                        # trying to improve the detection of the chessboard corners!
                        corners = cv2.cornerSubPix(gray,
                                                   corners,
                                                   (11, 11),  # size of kernel!
                                                   (-1, -1),
                                                   criteria)
                        cv2.drawChessboardCorners(img, (rows, columns), corners, ret)
                        for i, [corner] in enumerate(corners): # Check if enumeration is consistent
                            cv2.putText(img, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 0, 255), 1)
                        cv2.imshow(f'Chessboard corners; Camera: {self}', img)
                        key = cv2.waitKey(0)
                        if key & 0xFF == ord('s'):  # press s to switch the ordering of the corners
                            cv2.destroyWindow(f'Chessboard corners; Camera: {self}')
                            corners = corners[::-1]
                            # drawing the new corners
                            cv2.drawChessboardCorners(img_old, (rows, columns), corners, ret)
                            for i, [corner] in enumerate(corners):  # Check if enumeration is consistent
                                cv2.putText(img_old, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                                            (0, 0, 255), 1)
                            cv2.imshow(f'Chessboard corners; Camera: {self}', img_old)
                            cv2.waitKey(0)

                        cv2.destroyWindow(f'Chessboard corners; Camera: {self}')

                        objpoints.append(objp)
                        imgpoints.append(corners)

        # Camera resolution
        success, img = self.read()
        height = img.shape[0]
        width = img.shape[1]


        mtx = np.zeros((3,3))
        dist = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in imgpoints]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in imgpoints]
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC   + cv2.fisheye.CALIB_FIX_SKEW
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                np.expand_dims(np.asarray(objpoints), -2),
                imgpoints,
                gray.shape[::-1],
                mtx,
                dist,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )

        optimized_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            mtx, dist, img.shape[:2][::-1], np.eye(3), balance=0.5)


        print('rmse:', ret)
        print('camera matrix:\n', mtx)
        print('optimized camera matrix:\n', optimized_camera_matrix)
        print('distortion coeffs:\n', dist)

        self.fisheye_camera_matrix = mtx
        self.fisheye_optimized_camera_matrix = optimized_camera_matrix
        self.fisheye_distortion = dist
        self.fisheye = True

        # Updating the camera_configuration.json file
        self.set_config("fisheye_camera_matrix", self.fisheye_camera_matrix)
        self.set_config("fisheye_distortion", self.fisheye_distortion)
        self.set_config("fisheye_optimized_camera_matrix", self.fisheye_optimized_camera_matrix)
        self.set_config("fisheye", self.fisheye)

        # closing the window after calibration
        cv2.destroyWindow(f'Camera: {self}')
        self.calibrated = True


