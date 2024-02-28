import cv2
import dlib
import mediapipe as mp
import numpy as np
import time
from cvzone.FaceMeshModule import FaceMeshDetector



class legacy_faceDetection():
    """
    Callable class that detects faces in images.
    In this case the point between the eyes is used as the point location of the face.

    This can be done because cvzone detects FaceMeshes including eyes.
    """
    def __init__(self):
        self.detector = FaceMeshDetector(maxFaces=1, staticMode=True, minDetectionCon=0.1)

    def __call__(self, img):
        img, faces = self.detector.findFaceMesh(img, draw=False)
        # face landmarks 145 and 374 should be the right and the left eye. The position between these points is
        # calculated directly
        faces = [(np.array(face[145]) + np.array(face[374]))/2 for face in faces]
        return faces


class HaarrCascadeFaceDetection():
    """
    Callable class that detects faces in images.
    In this version I use a Haar Cascade Classifier from open CV

    returns list of face coordinates (one coordinate per face)
    [[342, 756], [...],...]
    """
    def __init__(self):
        """ Check https://www.youtube.com/watch?v=j27xINvkMvM"""
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        self.scale_factor = 1.2
        self.minNeighbors = 1
        self.minSize = None
        self.maxSize = None



    def __call__(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return self.face_detector.detectMultiScale(   frame,
                                                      scaleFactor=self.scale_factor,
                                                      minNeighbors=self.minNeighbors,
                                                      minSize=self.minSize,
                                                      maxSize=self.maxSize)


class hogfaceDetection():
    """
    Callable class that detects faces in images.

    returns list of face coordinates (one coordinate per face)
    [[342, 756], [...],...]
    """
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()

    def __call__(self, frame):
        face =  self.face_detector(frame)[0]
        face = str(face)
        face = face.split(' ')

        face = [[int(face[0].strip('[(,])')), int(face[1].strip('[(,])'))],
                [int(face[2].strip('[(,])')), int(face[3].strip('[(,])'))]]

        return face

class faceDetection():

    """
    Detects all faces in an Image and returns them as a list [[x,y], [x1, y1] ...]

    This faceDetection version is based on BlazeFace from mediapipe
    """
    def __init__(self, eye_detection=False):

        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)
        self.use_eye_detection = eye_detection

    def _eye_detection(self, face_cutout, cutout_corner, eye):
        """raises Error if no Eye is detected, this error will be caught in the call method.
            eye is either right, left or middle
        """
        raise Exception("No Eye detected!")

    def _approximate_eye(self, face_cutout, cutout_corner, eye):
        """
        Fallback if _eye_detection does not succeed. It just estimates the eye position on average values.
        :param face_cutout:
        :return:
        """
        cutout_height, cutout_width = face_cutout.shape[:2]

        return [int(cutout_height/2) + cutout_corner[0], int(cutout_width/2) + cutout_corner[1]]


    def __call__(self, img, eye="right"):
        """

        :param img:
        :param eye:  "right", "left", or "middle"
        :return:
        """
        assert eye in ["right", "left", "middle"], "eye has to be 'right', 'left' or 'middle'"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        faces = self.face_detector.process(img)

        if faces.detections:
            faces = [[   int(face.location_data.relative_bounding_box.xmin*img_width),
                         int(face.location_data.relative_bounding_box.ymin*img_height),
                         int(face.location_data.relative_bounding_box.width*img_width),
                         int(face.location_data.relative_bounding_box.height*img_height)] for face in faces.detections]

            # trying to detect eyes:
            eyes = []
            for i, face in enumerate(faces):
                cutout_corner = face[:2]
                face_cutout = img[face[1]:(face[1] + face[2]), face[0]: face[0] + face[3]]

                try:
                    eyes.append(self._eye_detection(face_cutout, cutout_corner, eye))
                except Exception:
                    eyes.append(self._approximate_eye(face_cutout, cutout_corner, eye))

            return eyes


        else:
            return []


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

    face_detected = False
    last_face_pos = []
    search_increase = 0.2

    stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    fd = faceDetection()

    while True:
        start_time = time.time()
        frame = stream.read()[1]
        print(f'Time for frame_read {time.time()-start_time}')

        face_detection_time = time.time()
        try:
            faces = fd(frame)
        except Exception as E:
            print('Face_detection failed', E)
            continue

        print(f'Time for face_detection{time.time()-face_detection_time}')
        print(faces)
        if len(faces) > 0:
            face_detected = True
            for face in faces:
                last_face_pos  = face
                cv2.circle(frame, (face[0], face[1]), radius=10, color=(0, 0, 0), thickness=10)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

        else:
            face_detected = False