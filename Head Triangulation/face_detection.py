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
    def __init__(self, use_eye_detection=True):

        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                                       min_detection_confidence=0.1)
        self.face_detector_fast = mp.solutions.face_detection.FaceDetection(model_selection=0,
                                                                            min_detection_confidence=0.4)
        self.use_eye_detection = use_eye_detection
        self.previous_faces = {}
        self.eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.scale_factor = 1.2
        self.minNeighbors = 1
        self.minSize = None
        self.maxSize = None


    def _eye_detection(self, face_cutout, cutout_corner, eye):
        """raises Error if no Eye is detected, this error will be caught in the call method.
            eye is either right, left or middle
        """
        if self.use_eye_detection:
            # To do cut out eye region before eye detection
            eye_detection_time = time.time()
            gray_cutout = cv2.cvtColor(face_cutout, cv2.COLOR_BGR2GRAY)
        
            faces = self.eye_detector.detectMultiScale(gray_cutout)[0]
            print('faces', faces, time.time() - eye_detection_time)
            return [faces[0] + int(faces[2]/2) + cutout_corner[0], faces[1] + int(faces[2]/2) + cutout_corner[1]]
        else:
            raise Exception("No Eye detected!")

    def _approximate_eye(self, face_cutout, cutout_corner, eye):
        """
        Fallback if _eye_detection does not succeed. It just estimates the eye position on average values.
        :param face_cutout:
        :return:
        """
        cutout_height, cutout_width = face_cutout.shape[:2]

        return [int(cutout_height/2) + cutout_corner[0], int(cutout_width/2) + cutout_corner[1]]

    def _create_face_patches(self, previous_face_regions, increase=2, min_res=[128, 128], img=None):
        """
        :param previous_face_regions:   Regions in which in the previous call Faces were detected
        :param increase:                Faktor by which the size of the original face regions is increased
        :param min_res:                 Minimal resolution, for the fast face detection model, this resolution will not
                                        be undercut for the face cutouts
        :return:                        Image cutout landmarks
        """
        face_centers = [[int(face_coords[2]/2) + face_coords[0], int(face_coords[3]/2) + face_coords[1]]
                        for face_coords in previous_face_regions]

        img_copy = np.array(img)
        side_relation_min_res = min_res[0] / min_res[1]

        cutout_resolutions = [[int(x*increase), int(y * increase)] for _,_, x, y in previous_face_regions]
        # replacing small cutout resolutions wiht the min_res
        cutout_resolutions = [res if res[0] > min_res[0] or res[1] > min_res[1] else min_res
                              for res in cutout_resolutions]
        # checking if side_relation of cutout_resolutions is correct
        cutout_resolutions = [res if res[0]/res[1] == side_relation_min_res else
                              [res[1] * side_relation_min_res, res[1]] for res in cutout_resolutions]

        # creating the cutout coordinates in the format of the face_predictions
        cutout_landmarks = [[int(abs(center[0] - res[0]/2)), int(abs(center[1] - res[1]/2)), int(abs(res[0])), int(abs(res[1]))]
                            for center, res in zip(face_centers, cutout_resolutions)]

        return  cutout_landmarks


    def __call__(self, img, camera=0, eye="right"):
        """

        :param img:
        :param eye:  "right", "left", or "middle"
        :param camera: indice which camera the image is from (important for roi detection)
        :return:
        """
        assert eye in ["right", "left", "middle"], "eye has to be 'right', 'left' or 'middle'"
        # loading previous face_positions
        try:
            previous_faces = self.previous_faces[camera]
        except KeyError: # No image from camera processed yet
            previous_faces = []


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        if len(previous_faces) == 0: # No previous face detection, using slow but accurate detection on whole image
            faces = self.face_detector.process(img)
            if faces.detections:
                faces_detected = True
            else:
                faces_detected = False


            if faces_detected:
                # !!! Faces are lists coded like [x_corner_of_cutout, y_corner_of_cutout, x_size_of_cutout, y_size_of_cutout]
                faces = [[   int(face.location_data.relative_bounding_box.xmin*img_width),
                             int(face.location_data.relative_bounding_box.ymin*img_height),
                             int(face.location_data.relative_bounding_box.width*img_width),
                             int(face.location_data.relative_bounding_box.height*img_height)] for face
                                                                                                    in faces.detections]

                # setting the previous faces for the next call
                self.previous_faces[camera] = faces

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
                # resetting the previous faces for next function call
                self.previous_faces[camera] = []
                return []

        # Faster Face detection algorithm on face patches based on previous face detections
        else:
            # creating face_patches
            face_cutout_landmarks = self._create_face_patches(previous_faces, img=img)

            face_offsets = [[x, y] for x, y,_, _ in face_cutout_landmarks]
            face_patches = [img[lm[1]:(lm[1] + lm[2]), lm[0]: (lm[0] + lm[3])] for lm in face_cutout_landmarks]
            predictions = [self.face_detector_fast.process(patch) for patch in face_patches]

            # Extracting the face bounding box values from the predictions
            faces = []
            for prediction, patch, (x_offset, y_offset) in zip(predictions, face_patches, face_offsets):
                patch_width, patch_height = patch.shape[:2]
                if prediction.detections:
                    for face in prediction.detections:
                        faces.append([int(x_offset + face.location_data.relative_bounding_box.xmin*patch_width),
                                      int(y_offset + face.location_data.relative_bounding_box.ymin*patch_height),
                                      int(face.location_data.relative_bounding_box.width*patch_width),
                                      int(face.location_data.relative_bounding_box.height*patch_height)])
            # setting the previous_faces for the new call
            self.previous_faces[camera] = faces

            # trying to detect eyes:
            eyes = []
            for i, face in enumerate(faces):
                cutout_corner = face[:2]
                face_cutout = img[face[1]:(face[1] + face[2]), face[0]: face[0] + face[3]]

                try:
                    eyes.append(self._eye_detection(face_cutout, cutout_corner, eye))
                except Exception as e:
                    print("Eye Detection failed, falling back to eye guessing", e)
                    eyes.append(self._approximate_eye(face_cutout, cutout_corner, eye))
            return eyes


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
        if len(faces) > 0:
            face_detected = True
            for face in faces:
                print('face in final loop', face)
                last_face_pos  = face
                cv2.circle(frame, (face[0], face[1]), radius=10, color=(0, 0, 0), thickness=10)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

        else:
            face_detected = False