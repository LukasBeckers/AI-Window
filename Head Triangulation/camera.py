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
        # loading the camera-configurations form camera_configurations.json
        if not self.load_config():
            print(f"No camera-configuration for name: {name} found!")
        self.stream = cv2.VideoCapture(self.stream)

    def read(self):
        return self.stream.read()

    def __str__(self):
        return self.name

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


        # storing the stream of the camera in a JSON format.
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

    def load_config(self):
        """
        If a camera with the same name was already configured, there will probably be a JSON file with
        configurations and calibrations for this setup, it will be loaded by this function.
        """
        try:
            with open('camera_configuration.json', 'r') as json_file:
                configurations = json.load(json_file)
                self.stream = configurations[self.name]["stream"]
                # trying to load attributes that are only added to the camera during calibration.
                try:
                    self.projection_error = configurations[self.name]["projection_error"]
                    self.camera_matrix = np.array(configurations[self.name]["camera_matrix"])
                    self.distortion = np.array(configurations[self.name]["distortion"])
                    self.tvecs = configurations[self.name]["tvecs"]
                    self.rvecs = configurations[self.name]["rvecs"]
                    self.calibrated = True
                except KeyError:
                    self.calibrated = False
            return True

        except FileNotFoundError:
            print("No camera_configuration.json file found!")
            return False


    def calibrate(self, rows=5, columns=7, n_images=5, scaling=0.01):
        """
        Thanks to: https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html

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
                        cv2.imshow(f'Chessboard corners; Camera: {self}', img)
                        cv2.waitKey(0)
                        cv2.destroyWindow(f'Chessboard corners; Camera: {self}')

                        objpoints.append(objp)
                        imgpoints.append(corners)

        # Camera resolution
        success, img = self.read()
        height = img.shape[0]
        width = img.shape[1]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        print('rmse:', ret)
        print('camera matrix:\n', mtx)
        print('distortion coeffs:', dist)
        print('Rs:\n', rvecs)
        print('Ts:\n', tvecs)

        self.projection_error = ret
        self.camera_matrix = mtx
        self.distortion = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

        # Updating the camera_configuration.json file
        self.set_config("projection_error", self.projection_error)
        self.set_config("camera_matrix", self.camera_matrix)
        self.set_config("distortion", self.distortion)
        self.set_config("rvecs", self.rvecs)
        self.set_config("tvecs", self.tvecs)
        # closing the window after calibration
        cv2.destroyWindow(f'Camera: {self}')
        self.calibrated = True

    def second_calibration(self, rows=7, columns=9, n_images=10, scaling=0.025):
            """
            Calibrates the camera, but on undistorted images using the distortion paramaters generated in the first
            calibration.
            :param rows:        Number of rows in chessboard-pattern.
            :param columns:     Number of columns in chessboard-pattern.
            :param scaling:     Realworld size of a chess-board square to scale the coordinate system.
                                I will try to keep all units in meters so a good initial value for this will be 0.01 or 1 cm

            :n_images:          Number of photos that will be taken for calibration.
            :return:
            """
            assert n_images > 0, "n_images must be larger than zero!"
            try:
                x = self.camera_matrix
                y = self.distortion
            except Exception:
                print("Camera not configured properly, could not load camera_matrix or distrortion. Maybe first ",
                      "calibration was not successfull.")

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

                if len(imgpoints) >= n_images:  # Processed enough pictures
                    break
                if success:
                    ##### Undistorting the image
                    h, w = img.shape[:2]
                    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix,
                                                                      self.distortion, (w, h), 1, (w, h))

                    img = cv2.undistort(img, self.camera_matrix, self.distortion, None,
                                        newcameramtx)

                    #####
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
                            cv2.imshow(f'Chessboard corners; Camera: {self}', img)
                            cv2.waitKey(0)
                            cv2.destroyWindow(f'Chessboard corners; Camera: {self}')

                            objpoints.append(objp)
                            imgpoints.append(corners)

            # Camera resolution
            success, img = self.read()
            height = img.shape[0]
            width = img.shape[1]

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
            print('rmse:', ret)
            print('camera matrix:\n', mtx)
            print('distortion coeffs:', dist)
            print('Rs:\n', rvecs)
            print('Ts:\n', tvecs)

            self.second_projection_error = ret
            self.second_camera_matrix = mtx
            self.second_distortion = dist
            self.second_rvecs = rvecs
            self.second_tvecs = tvecs

            # Updating the camera_configuration.json file
            self.set_config("second_projection_error", self.projection_error)
            self.set_config("second_camera_matrix", self.camera_matrix)
            self.set_config("second_distortion", self.distortion)
            self.set_config("second_rvecs", self.rvecs)
            self.set_config("second_tvecs", self.tvecs)
            # closing the window after calibration
            cv2.destroyWindow(f'Camera: {self}')
            self.calibrated = True

