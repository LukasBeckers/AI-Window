import cv2
import dlib
import mediapipe as mp
import numpy as np
import time


class mpFaceDetector():
    def __init__(self, model_selection, min_confidence, eye_box_relative_size=0.4):
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=model_selection,
                                                                  min_detection_confidence=min_confidence)
        self.box_size = eye_box_relative_size

    def __call__(self, img, eye=None):

        if eye is not None:
            eye = {"right": [0],
                   "left": [1],
                   "middle": [0, 1]}[eye]


        img_height, img_width = img.shape[:2]

        predictions = self.detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        bounding_boxes = []
        faces = []
        eyes = []
        eye_boxes = []
        # Extracting results form the predictions
        if predictions.detections:
            for face in predictions.detections:
                # !!! Faces are lists coded like [x_corner_of_cutout, y_corner_of_cutout, x_size_of_cutout, y_size_of_cutout]
                box = [int(face.location_data.relative_bounding_box.xmin * img_width),
                       int(face.location_data.relative_bounding_box.ymin * img_height),
                       int(face.location_data.relative_bounding_box.width * img_width),
                       int(face.location_data.relative_bounding_box.height * img_height)]

                bounding_boxes.append(box)
                faces.append([box[0] + box[2]/2, box[1] + box[3]/2])

                # extracting the eye_position
                if eye is not None:
                    if len(eye) == 1:
                        # Single eye
                        eye_position = [face.location_data.relative_keypoints[eye[0]].x * img_width,
                                                   face.location_data.relative_keypoints[eye[0]].y * img_height]
                        eyes.append(eye_position)

                        # Generating Eye-boxes for further eye-position refinement
                        eye_box_shape = [box[2] * self.box_size, box[3] * self.box_size]
                        eye_box_corner = [int(eye_position[0] - eye_box_shape[0]/2),
                                          int(eye_position[1]-eye_box_shape[1]/2)]
                        eye_box_corner.extend([int(x) for x in eye_box_shape])
                        eye_box = eye_box_corner
                        eye_boxes.append(eye_box)
                    else:
                        # Not a single eye but the middle between both eyes
                        x = [0, 0]
                        for i in eye:
                            x[0] += face.location_data.relative_keypoints[i].x * img_width
                            x[1] += face.location_data.relative_keypoints[i].y * img_height
                        x[0] /= len(eye)
                        x[1] /= len(eye)
                        eyes.append(x)

        output = {
            "faces": faces,
        }

        if len(bounding_boxes) > 0:
            output["bounding_boxes"] = bounding_boxes
        if len(eyes) > 0:
            output["eyes"] = eyes
        if len(eye_boxes) > 0:
            output["eye_boxes"] = eye_boxes

        return output

class faceDetection():

    """
    Detects all faces in an Image and returns them as a list [[x,y], [x1, y1] ...]

    This faceDetection version is build modular and can use different Face detection algorithms, which have to be passed
    as objects or functions to the faceDetection class during the __init__ call.

    This class allows you to pass a default and a fast face-detection algorithm.
    the thought behind this is, that the slow (but more sensitive)algorithm will be used on the full camera frame
    to detect any face. If faces are detected, only the region of the detected face, which was used in the downstream
    tasks (indces of faces to be tracked can be passed in the __call__ method) is analysed by the fast face-detection
    algorithm in this call, reducing latency.

    This face detection class also allows eye-detection based on the previous face-detection.
    The eye detection algorithm should also be passed as object or function to the __init__ call
    """
    def __init__(self, face_detector, face_detector_fast=None, eye_detector=None):
        """
        A face_detector or face_detector_fast object should be callable, take the following arguments:
        'img' = input image or patch
        'eye' = Some models also detect eyes while detecting faces, this param should indicate
                which eye should be detected possible values are: "right", "left", "middle"
                The detector must accept this argument, even if it doesn't do anything.

        and it should return the results as a dict, results are allowed to be floats of pixel values:
        following values are allowed in the returns:
        {
        "bounding_boxes": [[x_corner_of_bounding_box, y_corner_of_bounding_box,
                            x_size_of_bounding_box, y_size_of_bounding_box],
                            [x_corner_of_bounding_box2, ...], ...]

        "faces":         [[x_position_of_face, y_position_of_face],
                          [x_position_of_face2, ...], ..]

        "eyes":          [[x_position_of_eye, y_position_of_eye],
                          [x_position_of_eye2, ...],...]

        "eye_boxes"       [[x_corner_of_bounding_box, y_corner_of_bounding_box,
                         x_size_of_bounding_box, y_size_of_bounding_box],
                         [x_corner_of_bounding_box2, ...], ...]
        }
        "faces" is obligatory, all other results are optional.
        the "eye_boxes" coordinates should be in relation to the passed full-frame or patch pixel-coordinates,
        not the newly detected face_bounding_box.

        the eye_detector should take as input:

         "img", a cutout version of the face or a cutout_version of the "eye" previously
        detected by the face_detection , the

        "eye" parameter ["right", "left", "middle"] and a

        "refinement" argument,
        which indicates, that the algorithm should only refine the eye-position previously detected by the
        face_detection_algorithm.

        it should also return the results as a dict all result values are allowed to be floats of pixel values:

        {
        "eye": [x_position, y_position]
        }

        :param face_detector:      Default face-detection algorithm
        :param face_detector_fast: Fast face-detection algorithm, for contineous tracting of a face (optional)
        :param eye_detector:       Algorithm for eye-detection (optional)

        """
        self.face_detector = face_detector
        self.face_detector_fast = face_detector_fast
        self.eye_detector = eye_detector

        # stores bounding-box values of previous __call__ predictions
        self.previous_bounding_boxes = {}
        self.previous_eye_position_in_relation_to_bounding_box = {}

    def _eye_detection(self, img, predictions, eye):
        """raises Error if no Eye is detected, this error will be caught in the call method.
            eye is either right, left or middle
        """
        return predictions

    def _approximate_eyes(self, predictions, camera):
        """
        Fallback if _eye_detection does not succeed. It just estimates the eye position on average values.
        :param face_cutout:
        :return:
        """

        return predictions
    def _refine_eyes(self, img, predictions):
        for eye_box in predictions['eye_boxes']:
            eye_cutout = img[eye_box[1]:(eye_box[1] + eye_box[2]), eye_box[0]: (eye_box[0] + eye_box[3])]
            eye_cutout = cv2.resize(eye_cutout, [eye_cutout.shape[0] * 10, eye_cutout.shape[1] * 10])
            cv2.imshow('eye_cutout', eye_cutout)
            cv2.waitKey(1)

        return predictions

    def _create_face_patches(self, previous_face_regions, increase=2, min_res=[128, 128]):
        """
        :param previous_face_regions:   Regions in which in the previous call Faces were detected
        :param increase:                Faktor by which the size of the original face regions is increased
        :param min_res:                 Minimal resolution, for the fast face detection model, this resolution will not
                                        be undercut for the face cutouts
        :return:                        Image cutout landmarks
        """

        face_centers = [[int(face_coords[2]/2) + face_coords[0], int(face_coords[3]/2) + face_coords[1]]
                        for face_coords in previous_face_regions]
        side_relation_min_res = min_res[0] / min_res[1]

        cutout_resolutions = [[int(x*increase), int(y * increase)] for _,_, x, y in previous_face_regions]
        # replacing small cutout resolutions with the min_res
        cutout_resolutions = [res if res[0] > min_res[0] or res[1] > min_res[1] else min_res
                              for res in cutout_resolutions]
        # checking if side_relation of cutout_resolutions is correct
        cutout_resolutions = [res if res[0]/res[1] == side_relation_min_res else
                              [res[1] * side_relation_min_res, res[1]] for res in cutout_resolutions]

        # creating the cutout coordinates in the format of the face_predictions
        cutout_landmarks = [[int(abs(center[0] - res[0]/2)), int(abs(center[1] - res[1]/2)), int(abs(res[0])), int(abs(res[1]))]
                            for center, res in zip(face_centers, cutout_resolutions)]
        return  cutout_landmarks

    def _patch_prediction(self, img, eye, camera, tracked_faces):
        # sorting the previous bounding boxes
        if tracked_faces is not None:
            previous_bounding_boxes = [box for i, box in enumerate(self.previous_bounding_boxes[camera]) if i in
                                       tracked_faces]
        else:
            previous_bounding_boxes = self.previous_bounding_boxes[camera]

        # creating face_patches
        bounding_box_landmarks = self._create_face_patches(previous_bounding_boxes)
        bounding_box_offsets = [[x, y] for x, y, _, _ in bounding_box_landmarks]
        face_patches = [img[lm[1]:(lm[1] + lm[2]), lm[0]: (lm[0] + lm[3])] for lm in bounding_box_landmarks]

        # predicting and combining the predictions of all patches
        predictions = [self.face_detector_fast(patch, eye) for patch in face_patches]

        # merging the predictions of all patches and applying the offsets.
        merged_predictions = {}
        for pred, (x_offset, y_offset) in zip(predictions, bounding_box_offsets):
            for key, value in  pred.items():
                # applying the offsets to convert from patch pixel-values to img_pixel values
                value_new = []
                for v in value:
                    v[0] += x_offset
                    v[1] += y_offset
                    value_new.append(v)
                value = value_new
                #adding to the merged predictions
                if key in merged_predictions:
                    merged_predictions[key] += value
                else:
                    merged_predictions[key] = value
        predictions = merged_predictions

        # setting the previous bounding_boxes for the next call
        if "bounding_boxes" in predictions.keys():
            self.previous_bounding_boxes[camera] = predictions["bounding_boxes"]
        else:
            # resetting the previous bounding_boxes for next function call
            self.previous_bounding_boxes[camera] = []

        return predictions

    def _fullframe_prediction(self, img, eye, camera):
        predictions = self.face_detector(img, eye)

        # setting the previous bounding_boxes for the next call
        if "bounding_boxes" in predictions.keys():
            self.previous_bounding_boxes[camera] = predictions["bounding_boxes"]
        else:
            # resetting the previous bounding_boxes for next function call
            self.previous_bounding_boxes[camera] = []

        return predictions

    def __call__(self, img, camera=0, tracked_faces=None, eye=None, refinement=None):
        """

        :param img:         Image from camera-read.
        :param camera:      Index which camera the image is from (important for roi detection)
        :param eye:         If either the face-detection algorithm can detect eyes or a separate eye-detection
                            algorithm is passed, this parameter indicates which eye should be tracked:
                            "right", "left", "middle"
        :param refinement:  Indicates, if the eye-position should be predicted from a face cut out or if the
                            eye-position should just be refined based on a cutout of the eye_which is tracked.
                            either True, False the if the face_detection algorithm returns an "eye_boxes"
        :param tracked_faces: List of indices of faces from last prediction which should be tracked in this prediction,
                              None tracks all faces.

        :return:    List of face/eye pixel-positions (float)

        """

        assert eye in ["right", "left", "middle", None], "eye has to be 'right', 'left', 'middle' or None"

        # loading previous face_positions based on the camera-index
        try:
            previous_bounding_boxes = self.previous_bounding_boxes[camera]
        except KeyError: # No image from this camera processed yet
            previous_bounding_boxes = []

        # Full frame prediction using slow_detector
        if len(previous_bounding_boxes) == 0 or self.face_detector_fast is None:
            predictions = self._fullframe_prediction(img, eye, camera)
        # Faster Face detection algorithm on face patches based on previous face detections
        else:
            predictions = self._patch_prediction(img, eye, camera, tracked_faces)

        # Detecting or refining eye-position
        if self.eye_detector is not None:
            try:
                # Possible refinement branch
                if "eye_boxes" in predictions.keys():
                    predictions = self._refine_eyes(img, predictions)
                # Eye localization from face-patch
                elif "bounding_boxes" in predictions.keys():
                    predictions = self._eye_detection(img, predictions, eye)
            except Exception as exception:
                print("Eye detection or refinement failed, falling back to eye-estimation", exception)
                predictions = self._approximate_eyes(predictions, camera)

        if "eyes" in predictions.keys():
            return predictions["eyes"]
        else:
            return predictions["faces"]


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
    #stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    fd = faceDetection(face_detector=mpFaceDetector(model_selection=1, min_confidence=0.1),
                       face_detector_fast=mpFaceDetector(model_selection=0, min_confidence=0.4),
                       eye_detector=1)

    while True:
        start_time = time.time()
        frame = stream.read()[1]
        print(frame.shape)
        print(f'Time for frame_read {time.time()-start_time}')

        face_detection_time = time.time()

        faces = fd(frame, eye="right", tracked_faces=[0])
        print(f'Time for face_detection{time.time()-face_detection_time}')
        if len(faces) > 0:
            face_detected = True
            for face in faces:
                print('face in final loop', face)
                last_face_pos  = face
                cv2.circle(frame, (int(face[0]), int(face[1])), radius=1, color=(0, 0, 0), thickness=30)
            frame = cv2.resize(frame, (1280, 960))
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

        else:
            frame = cv2.resize(frame, (1280, 960))
            cv2.imshow('frame', frame)
            cv2.waitKey(1)