import threading
import json
import time

import cv2
from scipy import linalg
from camera import Camera
import matplotlib.pyplot as plt
import subprocess
from face_detection import *


class faceTriangulation():
    def read_cameras_synchronized(self):
        """
        Reads all cameras of this face Trianuglation set up and returns them semi synchronized.
        :return: list of camera frames
        """
        for _ in range(1):
            for camera in self.cameras:
                camera.stream.grab()
        images = [camera.stream.retrieve()[1]for camera in self.cameras]
        return images

    def stero_calibrate(self, rows=7, columns=9, n_images=6, scaling=0.025):
        """
        Calibrates the cameras for stereovision via several chess-board images.
        The calibration function should be automated in the future by automatic chess-board detection

        """

        for camera in self.cameras:
            if not camera.calibrated:
                print(f"Camera {camera} is not calibrated please calibrate all cameras first!")
                return

        # open cv can only detect inner corners, so reducing the rows and columns by one
        rows -= 1
        columns -= 1

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
            img_old = np.array(img1)
            success2, img2 = self.cameras[1].read()

            # counting the number of successful detections
            cv2.putText(img1, f'{len(imgpoints_1)}', (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
            cv2.putText(img2, f'{len(imgpoints_2)}', (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

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
                        for i, [corner] in enumerate(corners1):
                            cv2.putText(img1, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 0, 255), 1)
                        for i, [corner] in enumerate(corners2):
                            cv2.putText(img2, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 0, 255), 1)
                        cv2.imshow(f'Detection {self.cameras[0]}', img1)
                        cv2.imshow(f'Detection {self.cameras[1]}', img2)
                        key = cv2.waitKey(0)

                        if key & 0xFF == ord('s'):  # press s to switch ordering of img1
                            cv2.destroyWindow(f'Detection {self.cameras[0]}')
                            corners1 = corners1[::-1]
                            # drawing the new corners
                            cv2.drawChessboardCorners(img_old, (rows, columns), corners1, corner_success1)
                            for i, [corner] in enumerate(corners1):
                                cv2.putText(img_old, f'{i}', (int(corner[0]), int(corner[1])),
                                            cv2.FONT_HERSHEY_COMPLEX,
                                            1,
                                            (0, 0, 255), 1)
                            cv2.imshow(f'Detection {self.cameras[0]}', img_old)
                            cv2.waitKey(0)
                            # storing the realworld/ image-space coordinate pairs
                            objpoints.append(objp)
                            imgpoints_1.append(corners1)
                            imgpoints_2.append(corners2)
                            cv2.destroyWindow(f'Detection {self.cameras[0]}')
                            cv2.destroyWindow(f'Detection {self.cameras[1]}')

                        # press any other key (except esc) to use the detection
                        elif key > 0:
                            # storing the realworld/ image-space coordinate pairs
                            objpoints.append(objp)
                            imgpoints_1.append(corners1)
                            imgpoints_2.append(corners2)
                            cv2.destroyWindow(f'Detection {self.cameras[0]}')
                            cv2.destroyWindow(f'Detection {self.cameras[1]}')
        # prerform stereo calibration on accumulated objectpoints
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        if self.cameras[0].fisheye:
            imgpoints_1 = np.array(imgpoints_1)
            imgpoints_1 = np.swapaxes(imgpoints_1, axis1=1, axis2=2)
            imgpoints_2 = np.array(imgpoints_2)
            imgpoints_2 = np.swapaxes(imgpoints_2, axis1=1, axis2=2)
            objpoints = np.array(objpoints)
            objpoints = np.expand_dims(objpoints, axis=1)
            res = cv2.fisheye.stereoCalibrate(objpoints,
                                                                          imgpoints_1,
                                                                          imgpoints_2,
                                                                          self.cameras[0].fisheye_camera_matrix,
                                                                          self.cameras[0].fisheye_distortion,
                                                                          self.cameras[1].fisheye_camera_matrix,
                                                                          self.cameras[1].fisheye_distortion,
                                                                          (width, height),
                                                                          criteria=criteria,
                                                                          flags=stereocalibration_flags)
            ret, CM1, dist1, CM2, dist2, R, T = res[:7]
        else:
            ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                          imgpoints_1,
                                                                          imgpoints_2,
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

        cv2.destroyAllWindows()
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

    def DLT(self, P1, P2, point1, point2):
        A = [point1[1] * P1[2, :] - P1[1, :],
             P1[0, :] - point1[0] * P1[2, :],
             point2[1] * P2[2, :] - P2[1, :],
             P2[0, :] - point2[0] * P2[2, :]
             ]
        A = np.array(A).reshape((4, 4))
        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices=False)

        return Vh[3, 0:3] / Vh[3, 3]

    def triangulate_face(self, img1, img2):
        """
        :param img1 & img2:  Rectified images from both cameras
        :return:
        """
        # detection both faces in parallel
        face1 = self.detect_face(img1, camera=0)
        face2 = self.detect_face(img2, camera=1)

        # rotation + translation matrix for camera 0 is identity.
        RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
        if self.cameras[0].fisheye:
            P1 = self.cameras[0].current_camera_matrix @ RT1  # projection matrix for C1
        else:
            P1 = self.cameras[0].current_camera_matrix @ RT1  # projection matrix for C1

        # Rotation x translation matrix for camera 1 is the rotation and translation matrix obtained by
        # the stereo-calibration.
        RT2 = np.concatenate([self.rotation_matrix, self.translation_matrix], axis=-1)
        if self.cameras[1].fisheye:
            P2 = self.cameras[1].current_camera_matrix @ RT2
        else:
            P2 = self.cameras[1].current_camera_matrix @ RT2

        # Using the previous value for face1, face2 and face_coordinates, if no face was detected.
        try:
            # face_detection has to be finished before the triangulation
            if face1 and face2:
                face_coordinates = self.DLT(P1, P2, face1, face2)
            else: # if one frame did not contain a face, the previous detections are returned
                return self.face1, self.face2
        except IndexError: # Index error occurs if no face was detected
            return self.face1, self.face2
        self.face_coordinates = face_coordinates
        if len(face1) == 2:
            self.face1 = face1
        if len(face2) == 2:
            self.face2 = face2

        face1 = self.face1
        face2 = self.face2

        return face1, face2

        # Function to compute the transformation matrix

    def compute_transform_matrix(self, points_A, points_B):
        """Calculates the transform matrix to transform any point of one coordinatesystem (3d) to another"""
        # Assuming points_A and points_B are numpy arrays with shape (N, 3)
        # where N is the number of points (N >= 3),
        # Reshaping the points
        points_A = np.array(points_A)
        points_B = np.array(points_B)
        points_A = points_A.T
        points_B = points_B.T

        # Calculate the centroids of both point sets
        centroid_A = np.mean(points_A, axis=1, keepdims=True)
        centroid_B = np.mean(points_B, axis=1, keepdims=True)

        # Compute the centered point sets
        centered_A = points_A - centroid_A
        centered_B = points_B - centroid_B

        # Compute the covariance matrix
        covariance_matrix = centered_A @ centered_B.T

        # Perform Singular Value Decomposition
        U, _, Vt = np.linalg.svd(covariance_matrix)

        # Calculate the rotation matrix
        rotation_matrix = Vt.T @ U.T

        # Calculate the translation vector
        translation_vector = centroid_B - rotation_matrix @ centroid_A

        # Create the transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation_vector.flatten()
        return transform_matrix

    def transform_point(self, point, transform_matrix):
        """Transforms a point to a nother coordinate system (3d) based on a transform_matrix"""
        # point needs one extra dimension
        point = np.array([point])
        point_homogeneous = np.hstack((point, np.ones((1, 1))))
        new_point_homogeneous = transform_matrix @ point_homogeneous.T

        # Extract the transformed point in system B
        new_point = new_point_homogeneous[:3, :].T
        return new_point[0]

    def calibrate_display_center(self, rows=4, columns=5, points=[(0.485, 0.03, 0.0375), (0.485, 0.055, 0.0375), (0.485, 0.08, 0.0375), (0.485, 0.03, 0.0125), (0.485, 0.055, 0.0125), (0.485, 0.08, 0.0125),(0.485, 0.03, -0.0125), (0.485, 0.055, -0.0125), (0.485, 0.08, -0.0125),(0.485, 0.03, -0.0375), (0.485, 0.055, -0.0375), (0.485, 0.08, -0.0375)]):
        """
        Changes the coordinate system to a coordinate system based in the display center using the calibration device
        :param rows:
        :param columns:
        :param points:      Corresponding points in the target coorddinate system.
                            In case of the first calibration device, the detected points must follow the following
                            order in the image space:
                            2, 5, 8, 11
                            1, 4, 7, 10
                            0, 3, 6, 9
        :return:
        """
        points = np.array(points) #convert to meters

        # inner lines of a checkerboard are allways one less than the number of rows and columns
        rows -= 1
        columns -= 1

        # showing dots to show screen center for calibration device
        # !!!!! Richtig schlechter Code!!!!! Ändern! und intern über pygame regeln
        try:
            subprocess.Popen(["python", "show_dots.py"])
        except Exception as exception:
            print(f"Exception during subprocess call: {exception}")

        # Global variables to store the selected region points
        global selected_points_img1
        global selected_points_img2

        selected_points_img1 = []
        selected_points_img2 = []
        while True:
            # reading the frames form the cameras
            success1, img1 = self.cameras[0].read()
            img_old = np.array(img1)
            success2, img2 = self.cameras[1].read()

            img1 = self.cameras[0].undistort_image(img1)
            img2 = self.cameras[1].undistort_image(img2)

            if success1 and success2:

                def mouse_callback_img1(event, x, y, flags, param):
                    global selected_points_img1
                    if event == cv2.EVENT_LBUTTONDOWN:
                        print("Left mouse button pressed!", selected_points_img1)
                        selected_points_img1.append((x, y))


                def mouse_callback_img2(event, x, y, flags, param):
                    global selected_points_img2
                    if event == cv2.EVENT_LBUTTONDOWN:
                        print("Left mouse button pressed!", selected_points_img2)
                        selected_points_img2.append((x, y))


                cv2.namedWindow("img1")
                cv2.namedWindow("img2")
                # Set mouse callbacks
                cv2.setMouseCallback("img1", mouse_callback_img1)
                cv2.setMouseCallback("img2", mouse_callback_img2)

                cv2.imshow(f"img1", img1)
                cv2.imshow(f"img2", img2)

                key = cv2.waitKey(1)
                if key == 27:  # esc stops the calibration process
                    print("Aborting.")
                    return
                # press any key except esc to take the frame for calibration
                if key > 0:
                    # Wait for the user to select the region in both images
                    while len(selected_points_img1) < 2 or len(selected_points_img2) < 2:
                        key = cv2.waitKey(1)
                        if key == 27:  # esc stops the calibration process
                            print("Aborting.")
                            return
                    # Inpainting the selected regions
                    cv2.rectangle(img1, selected_points_img1[0], selected_points_img1[1], (0, 255, 0), 2)
                    cv2.imshow("img1", img1)
                    cv2.rectangle(img2, selected_points_img2[0], selected_points_img2[1], (0, 255, 0), 2)
                    cv2.imshow("img2", img2)
                    key = cv2.waitKey(0)
                    # Extract the selected regions from img1 and img2

                    offset1 = [min([selected_points_img1[0][0], selected_points_img1[1][0]]),
                                    min([selected_points_img1[0][1], selected_points_img1[1][1]])]
                    offset2 = [min([selected_points_img2[0][0], selected_points_img2[1][0]]),
                                    min([selected_points_img2[0][1], selected_points_img2[1][1]])]

                    img1 = img1[selected_points_img1[0][1]:selected_points_img1[1][1],
                               selected_points_img1[0][0]:selected_points_img1[1][0]]
                    img2 = img2[selected_points_img2[0][1]:selected_points_img2[1][1],
                               selected_points_img2[0][0]:selected_points_img2[1][0]]

                    #save copies of original  images
                    img1_old = np.array(img1)
                    img2_old = np.array(img2)
                    # Reset selected points
                    selected_points_img1 = []
                    selected_points_img2 = []
                    # Detecting the checkerboard corners
                    # for checkerboard detection

                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)
                    # gray1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    # very cool effect, but does not improve detection
                    # gray2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                    corner_success1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns),None)
                    corner_success2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns),None)
                    if corner_success2 and corner_success1:
                        cv2.destroyWindow(f'img1')
                        cv2.destroyWindow(f'img2')
                        # Refining the detection
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
                        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

                        # Showing the detection
                        cv2.drawChessboardCorners(gray1, (rows, columns), corners1, corner_success1)
                        cv2.drawChessboardCorners(gray2, (rows, columns), corners2, corner_success2)
                        for i, [corner] in enumerate(corners1): # Check if enumeration is consistent
                            cv2.putText(gray1, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 0, 255), 1)
                        for i, [corner] in enumerate(corners2):
                            cv2.putText(gray2, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                                        (0, 0, 255), 1)

                        cv2.imshow(f'Detection {self.cameras[0]}', gray1)
                        cv2.imshow(f'Detection {self.cameras[1]}', gray2)
                        key = cv2.waitKey(0)

                        if key & 0xFF == ord('s'):  # press s to switch corners ordering of img1
                            cv2.destroyWindow(f'Detection {self.cameras[0]}')
                            # reversing the order of the corners
                            corners1 = corners1[::-1]
                            # drawing the new corners
                            cv2.drawChessboardCorners(img1_old, (rows, columns), corners1, corner_success1)
                            for i, [corner] in enumerate(corners1):  # Check if enumeration is consistent
                                cv2.putText(img1_old, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX,
                                            1,
                                            (0, 0, 255), 1)
                            cv2.imshow(f'Detection {self.cameras[0]}', img1_old)
                            cv2.waitKey(0)
                        elif key > 0:  # press any other key to continue with the triangulation
                            # storing the realworld / image-space coordinate pairs
                            # rotation + translation matrix for camera 0 is identity.
                            RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
                            if self.cameras[0].fisheye:
                                P1 = self.cameras[0].fisheye_optimized_camera_matrix @ RT1  # projection matrix for C1
                            else:
                                P1 = self.cameras[0].optimized_camera_matrix @ RT1  # projection matrix for C1

                            # Rotation x translation matrix for camera 1 is the rotation and translation matrix obtained by
                            # the stereo-calibration.
                            RT2 = np.concatenate([self.rotation_matrix, self.translation_matrix], axis=-1)
                            if self.cameras[1].fisheye:
                                P2 = self.cameras[1].fisheye_optimized_camera_matrix @ RT2  # projection matrix for C1
                            else:
                                P2 = self.cameras[1].optimized_camera_matrix @ RT2  # projection matrix for C1

                            coordinates = []
                            # removing the scaling and adding the offset to get pixle coordinates in whole image
                            corners1 = [corner + offset1 for corner in corners1]
                            corners2 = [corner + offset2 for corner in corners2]

                            # triangulating the coordinates
                            for c1, c2 in zip(corners1, corners2):
                                coordinates.append(self.DLT(P1, P2, c1[0], c2[0]))
                            break
        print("Coordinates", coordinates)

        # calculating the transformation matrix to transform from camera coordiante system to display center coordiante system
        transform_matrix = self.compute_transform_matrix(coordinates, points)
        self.transform_matrix = transform_matrix
        self.display_center_calibrated = True
        self.set_config("transform_matrix", transform_matrix)

        # transform to display coordinatesystem
        coordinates = [self.transform_point(point, self.transform_matrix) for point in coordinates]
        #coordinates = points.tolist()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for x, y, z in coordinates:
            ax.scatter(x, y, z, c="red", s=8)

        # plotting the cameras
        # camera 1
        cam1_pos = self.transform_point([0,0,0], self.transform_matrix)
        ax.scatter(cam1_pos[0], cam1_pos[1], cam1_pos[2], c="green", s=15)

        # Camera 2
        cam2_pos = [-x[0] for x in self.translation_matrix]
        cam2_pos = self.transform_point(cam2_pos, self.transform_matrix)
        ax.scatter(cam2_pos[0], cam2_pos[1], cam2_pos[2], c="green", s=15)

        # Plotting the display center
        ax.scatter(0,0,0, c="blue", s=30)

        # add camera positions
        coordinates.append(cam1_pos)
        coordinates.append(cam2_pos)

        # add display center
        coordinates.append([0,0,0])
        x_coords = [c[0] for c in coordinates]
        y_coords = [c[1] for c in coordinates]
        z_coords = [c[2] for c in coordinates]

        ax.set_xlim([min(x_coords), max(x_coords)])
        ax.set_ylim([min(y_coords), max(y_coords)])
        ax.set_zlim([min(z_coords), max(z_coords)])
        ax.set_box_aspect([np.ptp(x_coords), np.ptp(y_coords), np.ptp(z_coords)])
        cv2.destroyAllWindows()
        plt.show()


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
        self.display_center_calibrated = False
        if not self.load_config():
            print(f"No SpatialFacePosition configuration named {self.name} found!")
        self.face_detection = face_detection

        self.handle_faces = handle_faces

        # Initializing the cameras
        self.cameras = [Camera(camera) for camera in self.cameras]
        self.face_coordinates = [1,0,0]
        self.face1 = [0, 0]
        self.face2 = [0, 0]

        #Initially reading the cameras
        current_frames = [None] * len(self.cameras)
        def read_camera(camera_index):
            camera = self.cameras[camera_index]
            camera.stream.grab()
            success, img =camera.stream.retrieve()
            current_frames[camera_index] = (success, img)

        camera_threads = []
        for i in range(len(self.cameras)):
            thread = threading.Thread(target=read_camera, args=(i, ))
            camera_threads.append(thread)
            thread.start()

        for thread in camera_threads:
            thread.join()
        self.camera_threads = camera_threads
        self.current_frames = current_frames

    def load_config(self):
        """
        If a configuration with the same name was already initialized, there will be a JSON file with
        configurations and calibrations for this setup, it will be loaded by this function.
        """
        try:
            with open('face_position_configurations.json', 'r') as json_file:
                configurations = json.load(json_file)
                self.cameras = configurations[self.name]["cameras"]
                # loading parameters that are only stored after stereo-calibration
                try:
                    self.rotation_matrix = configurations[self.name]["rotation_matrix"]
                    self.translation_matrix = configurations[self.name]["translation_matrix"]
                    self.stereo_calibrated = True
                except KeyError:
                    self.stereo_calibrated = False
                # loading the parameters that are only stored after display-center-calibration
                try:
                    self.transform_matrix = configurations[self.name]["transform_matrix"]
                    self.display_center_calibrated = True
                except KeyError:
                    self.display_center_calibrated = False

            return True

        except FileNotFoundError:
            print("No configurations.json file found!")
            return False

    def detect_face(self, img, camera):
        """
        Detects faces in an img using the face_detection attribute and uses the face_handeler to select the correct
        face from all detected faces.
        :param img:         Image from a camera
        :return:            face_coordinates
        """
        faces = self.face_detection(img, camera)
        face = self.handle_faces(faces)
        return face

    def triangulate_checkerboard(self, rows=7, columns=9):
        """
        Just for testing, triangulates all corners of a checkerboard in 3D space.
        After triangulation, the coordinates are shown in a 3D plot to check if they resemble a checkerboard.
        """
        # inner lines of a checkerboard are allways one less than the number of rows and columns
        rows -= 1
        columns -= 1

        # for checkerboard detection
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        while True:
            # reading the frames form the cameras
            success1, img1 = self.cameras[0].read()
            success2, img2 = self.cameras[1].read()


            img1 = self.cameras[0].undistort_image(img1)
            img2 = self.cameras[1].undistort_image(img2)


            if success1 and success2:
                # Displaying the images
                cv2.imshow(f"{self.cameras[0]}", img1)
                cv2.imshow(f"{self.cameras[1]}", img2)
                key = cv2.waitKey(1)
                if key == 27:  # esc stops the calibration process
                    print("Aborting.")
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
                        for i, [corner] in enumerate(corners1):
                            cv2.putText(img1, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                        for i, [corner] in enumerate(corners2):
                            cv2.putText(img2, f'{i}', (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

                        cv2.imshow(f'Detection {self.cameras[0]}', img1)
                        cv2.imshow(f'Detection {self.cameras[1]}', img2)
                        key = cv2.waitKey(0)

                        if key & 0xFF == ord('s'): # press s to skip due to bad prediction
                            cv2.destroyWindow(f'Detection {self.cameras[0]}')
                            cv2.destroyWindow(f'Detection {self.cameras[1]}')
                            continue
                        elif key > 0: # press any other key to continue with the triangulation
                            # storing the realworld/ image-space coordinate pairs
                            # rotation + translation matrix for camera 0 is identity.
                            RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
                            #P1 = self.cameras[0].camera_matrix @ RT1  # projection matrix for C1
                            # Testing with undistorted images
                            if self.cameras[0].fisheye:
                                P1 = self.cameras[0].fisheye_optimized_camera_matrix @ RT1  # projection matrix for C1
                            else:
                                P1 = self.cameras[0].optimized_camera_matrix @ RT1  # projection matrix for C1

                            # Rotation x translation matrix for camera 1 is the rotation and translation matrix obtained by
                            # the stereo-calibration.
                            RT2 = np.concatenate([self.rotation_matrix, self.translation_matrix], axis=-1)
                            #P2 = self.cameras[1].camera_matrix @ RT2
                            # Testing with undistorted images
                            if self.cameras[1].fisheye:
                                P2 = self.cameras[1].fisheye_optimized_camera_matrix @ RT2  # projection matrix for C1
                            else:
                                P2 = self.cameras[1].optimized_camera_matrix @ RT2  # projection matrix for C1

                            coordinates = []
                            for c1, c2 in zip(corners1, corners2):
                                coordinates.append(self.DLT(P1, P2, c1[0], c2[0]))
                            break
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for x, y, z in coordinates:
            ax.scatter(x, y, z, c="red", s=8)

        # plotting the cameras
        # camera 1
        ax.scatter(0, 0, 0, c="green", s=15)  # Adjust the size (s) to  make it larger

        # Camera 2
        cam2_pos = [-x[0] for x in self.translation_matrix]
        ax.scatter(cam2_pos[0], cam2_pos[1], cam2_pos[2], c="green", s=10)

        coordinates.append([0, 0, 0])
        coordinates.append(cam2_pos)

        x_coords = [c[0] for c in coordinates]
        y_coords = [c[1] for c in coordinates]
        z_coords = [c[2] for c in coordinates]

        ax.set_xlim([min(x_coords), max(x_coords)])
        ax.set_ylim([min(y_coords), max(y_coords)])
        ax.set_zlim([min(z_coords), max(z_coords)])
        ax.set_box_aspect([np.ptp(x_coords), np.ptp(y_coords), np.ptp(z_coords)])
        cv2.destroyAllWindows()
        plt.show()

    def __call__(self):
        """
        Reads both streams, loads just one frame and predicts the spatial position of the head in relation to the
        screen.

        :return:    Success face_pos img1 (with face annotations) img2 (with face annotations)
        """
        call_time = time.time()
        time_cam = time.time()
        for thread in self.camera_threads:
            thread.join()
        success1, img1 = self.current_frames[0]
        success2, img2 = self.current_frames[1]

        # renewing the threads
        def read_camera(camera_index):
            #camera = self.cameras[camera_index]
            #camera.stream.grab()
            #success, img = camera.stream.retrieve()
            #if success:
            #    img = camera.undistort_image(img)
            #self.current_frames[camera_index] = (success, img)
            self.current_frames = [self.cameras[0].read(), self.cameras[1].read()]

        self.camera_threads = []
        for i in range(len(self.cameras[:1])):
            thread = threading.Thread(target=read_camera, args=(i, ))
            self.camera_threads.append(thread)
            thread.start()

        # If successfullly read frames, start processing
        if success1 and success2:
            self.cameras[0].current_camera_matrix = self.cameras[0].optimized_camera_matrix
            self.cameras[1].current_camera_matrix = self.cameras[1].optimized_camera_matrix

            tri_time = time.time()
            # face1 and face2 are image coorinates of the face, the 3d face coordinates will be saved in self.face_coordinates
            face1, face2 = self.triangulate_face(img1, img2)
            tri_time = tri_time - time.time()

            transform_time = time.time()
            face_coordinates = self.face_coordinates

            # convert to display coordinates if display_center_calibrated it transforms the face_coordinates for the coorinate
            # system based in camera0 to a coordinate system based in the display center.
            if self.display_center_calibrated:
                face_coordinates = self.transform_point(face_coordinates, self.transform_matrix)
            print(f"""
            Time for whole call: {call_time - time.time()}, 
            time for triangulation: {tri_time}, 
            time for transformation: {time.time()-transform_time}""")
            return face_coordinates, img1, img2, face1, face2
        else:
            self.camera_threads = []
            for i in range(len(self.cameras)):
                thread = threading.Thread(target=read_camera, args=(i,))
                self.camera_threads.append(thread)
                thread.start()
            print(f'could not read all cameras')




if __name__ == "__main__":


    import cProfile
    import pstats

    spf = spatialFacePosition(name='Setup Aachen 1 1080')

    for i in range(10000):
        x = spf()

    def test_function():
        """
                if not self.load_configs():
            print(f"No camera-configuration for name: {name} found!")
        if self.resolution:
            self.stream = cv2.VideoCapture(self.stream, cv2.CAP_DSHOW)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[1])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[0])
        else:
            self.stream = cv2.VideoCapture(self.stream)

        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        :return:
        """
        stream = cv2.VideoCapture(0)
        stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        stream.set(cv2.CAP_PROP_FPS, 60)
        stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        stream2 = cv2.VideoCapture(1)
        stream2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        stream2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        stream2.set(cv2.CAP_PROP_FPS, 60)

        average_t = 0
        for x in range(20):

            t = time.time()
            ret, img = stream.read()
            #ret, img2 = stream2.read()
            if ret:
                print('time for stream read:', time.time() - t, ret, img.shape)
                if x > 2:
                    average_t += (time.time() - t)
            else:
                print('could not read cam')
        print('took on average', average_t/20)

    #test_function()


