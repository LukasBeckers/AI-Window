import cv2
import mediapipe as mp
import numpy as np
import time
from cvzone.FaceMeshModule import FaceMeshDetector



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
        print(faces)
        return faces

class checkerboardDetection():
    """
    Callable class that detects all checkerboard corners in an image, containing a 5x7 checkerboard

    This class is meant for debugging purposes, because the checkerboard-corner localization is much more precise than
    the facedetection.
    """
    def __init__(self):
        pass


    def __call__(self, img):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # Detecting the checkerboard corners
        corner_success, corners = cv2.findChessboardCorners(gray, (4, 6), None)
        if corner_success:
            # Refining the detection
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        else:
            corners = [[[]]]
        # unpacking the inner lists
        corners = [c[0] for c in corners]
        return corners


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



if __name__=="__main__":
    pass