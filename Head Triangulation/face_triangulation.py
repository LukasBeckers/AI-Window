import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from multiprocessing import get_context, TimeoutError
import numpy as np
import time
import os
import json
from camera import Camera
from spatial_face_position import *


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
    spatial_face_position = spatialFacePosition("Setup Aachen 1")
    if not spatial_face_position.stereo_calibrated:
        spatial_face_position.stero_calibrate()
    spatial_face_position.stero_calibrate()




