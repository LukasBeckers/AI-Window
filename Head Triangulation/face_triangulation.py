import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from multiprocessing import get_context, TimeoutError
import numpy as np
import time
import os
import json
import pickle as pk
from camera import Camera
from spatial_face_position import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def configure_spatial_face_positon(name,
                                   cameras,
                                   monitor_center=[0, 0, 0],
                                   monitor_normal=[0, 0, 1],
                                   monitor_rotation=[0, 1, 0]):
        """
        stores the configuration of a spatialFacePosition setup.
        :param name:            Name of this setup, using this name the configuration will be stored in the
                                face_position_configuration.json file.
        :param cameras:         Name of all cameras that will be used in this setup, the cameras must have a config
                                using this name in the camera_configurations.json.

        :param monitor_center:        [list] Vector pointing from the "main"-camera towards the monitor center. This
                                      vector is based on the main-cameras coordinate system.
        :param monitor_normal:        [list] Plane-normal of the monitor, based on the "main"-cameras coordinate system.
                                      This vector is based on the main-cameras coordinate system.
                                      It will also function as the x-axis in the screen coordinate system.
        :param monitor_rotation:      [list] Vector that is parallel to the lower edge of the screen. It is also based
                                      on the main-cameras coordinate system and will function as the z-coordinate in
                                      the screen coordinate system.

        """
        # all configurations are stored in a single json file separated by their setup-names
        try:
            with open('face_position_configurations.json', 'r') as json_file:
                configurations = json.load(json_file)
        except FileNotFoundError:
            configurations = {}

        # storing the general setup configurations
        configurations[name] = {
            "cameras": cameras,
            "monitor_center": monitor_center,
            "monitor_normal": monitor_normal,
            "monitor_rotation": monitor_rotation
        }

        # saving the dict as JSON
        with open('face_position_configurations.json', 'w') as json_file:
            json.dump(configurations, json_file)


def configure_camera(name, stream):
    """
    stores the initial configuration of a Camera setup.

    :param name:            Name of this camera, using this name the configuration will be stored in the
                            camera_configuration.json file.
    :param stream:          Number of the open cv stream that should be used for this camera-setup.

    """
    # all camera configurations are stored in a single json file separated by their setup-names
    try:
        with open('camera_configuration.json', 'r') as json_file:
            configurations = json.load(json_file)
    except FileNotFoundError:
        configurations = {}

    # storing the stream of the camera in a JSON format.
    configurations[name] = {
        "stream": stream
    }

    # saving the dict as JSON
    with open('camera_configuration.json', 'w') as json_file:
        json.dump(configurations, json_file)


def normalize(vector):
    """
    Normalizes a vector to an unit vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    else:
        return vector/norm



if __name__ == "__main__":
    #configure_camera("Camera0", 0)
    #configure_camera("Camera1", 1)
    #configure_spatial_face_positon("Setup Aachen 1", ["Camera0", "Camera1"])
    spatial_face_position = spatialFacePosition("Setup Aachen 1", face_detection=faceDetection())
    if not spatial_face_position.stereo_calibrated:
        spatial_face_position.stero_calibrate()
    #for camera in spatial_face_position.cameras:
           #camera.calibrate(scaling=0.025, n_images=10, rows=7, columns=9)
           #camera.second_calibration()
    #spatial_face_position.stero_calibrate(scaling=0.025, n_images=10, rows=7, columns=9)
    for camera in spatial_face_position.cameras:
        print("Internal Parameters")
        print(camera.camera_matrix)
        print(camera.distortion)
    #spatial_face_position.triangulate_checkerboard(rows=7, columns=9)



    face_coordinates = []
    frames = []

    if True:
        while True:
            coordinate, frame1, frame2, face1, face2 = spatial_face_position()
            print(frame1.shape)
            face_coordinates.append(coordinate)
            print(coordinate)
            cv2.circle(frame1, [int(face1[0]), int(face1[1])], 5, (255, 0, 0), -1)
            frames.append(frame1)
            cv2.circle(frame2, [int(face2[0]), int(face2[1])], 5, (255, 0, 0), -1)
            cv2.imshow("Show1", frame1)
            cv2.imshow("Show2", frame2)
            k = cv2.waitKey(1)
            if k > 0:
                break


        with open('coordinates.pickle', "wb") as f:
            pk.dump(face_coordinates, f)
        # Set the output video file name and parameters
        try:
            output_video_path = 'output_video.mp4'
            fps = 30.0  # Frames per second
            frame_size = frames[0].shape[:2][::-1]  # Specify the width and height of each frame
            print("Frame Size", frame_size)
            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID', 'MJPG', etc.
            out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

            # Write each frame to the video file
            for frame in frames:
                out.write(frame)

            # Release the VideoWriter object
        finally:
            out.release()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        connections = [[a, b] for a, b in zip(range(len(face_coordinates)-1), range(1, len(face_coordinates)))]

        face_coordinates = np.array(face_coordinates)
        for _c in connections:
            ax.plot(xs = [face_coordinates[_c[0],0], face_coordinates[_c[1],0]], ys = [face_coordinates[_c[0],1], face_coordinates[_c[1],1]], zs = [face_coordinates[_c[0],2], face_coordinates[_c[1],2]], c = 'red')

        # plotting the cameras
        # camera 1
        ax.scatter(0, 0, 0, c="green", s=15)  # Adjust the size (s) to make it larger

        # Camera 2
        cam2_pos = [x[0] for x in spatial_face_position.translation_matrix]
        ax.scatter(cam2_pos[0], cam2_pos[1], cam2_pos[2], c="green", s=10)
        face_coordinates = face_coordinates.tolist()

        face_coordinates.append([0,0,0])
        face_coordinates.append(cam2_pos)

        x_coords = [c[0] for c in face_coordinates]
        y_coords = [c[1] for c in face_coordinates]
        z_coords = [c[2] for c in face_coordinates]

        ax.set_xlim([min(x_coords), max(x_coords)])
        ax.set_ylim([min(y_coords), max(y_coords)])
        ax.set_zlim([min(z_coords), max(z_coords)])
        ax.set_box_aspect([np.ptp(x_coords), np.ptp(y_coords), np.ptp(z_coords)])
        plt.show()




