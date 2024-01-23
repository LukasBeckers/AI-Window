import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from multiprocessing import get_context, TimeoutError
import numpy as np
import time


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
        faces = [[(np.array(face[145]) + np.array(face[374]))/2] for face in faces]
        return faces


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

    def calibrate(self, name: str = 'unknown'):
        """
        Calibrates the cameras for stereovision via several chess-board images.
        The calibration function should be automated in the future by automatic chess-board detection

        :param name:     name of this setup (calibration parameters will be saved using this name)
        :return:         Calibration matrix?
        """
        stream_1 = self.cap1
        stream_2 = self.cap2
        successes = 0
        while True:

            success1, image1 = stream_1.read()
            success2, image2 = stream_2.read()

            cv2.imshow('Image1', image1)
            cv2.imshow('Image2', image2)

            k = cv2.waitKey(50)

            chessboard_size = (6, 4)
            n_images = 4
            frameSize = image2.shape[:2]

            print(frameSize)
            # Exit if esc is pressed
            if k == 27:
                break
            # Save Images if s is pressed
            elif k == ord('s'):

                # Termination criteria
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

                # Prepare object points, like (0,0,0), (1,0,0), (2,0,0)...
                objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0: chessboard_size[0], 0: chessboard_size[1]].T.reshape(-1, 2)

                # Arrays to store object points and image points from all the images.
                objpoints = []  # 3d points in real world space
                imgpoints1 = []  # 2d points in image plane
                imgpoints2 = []  # 2d points in image plane.

                gray1, gray2 = cv2.cvtColor(image1, cv2.COLOR_BGRA2GRAY), cv2.cvtColor(image2, cv2.COLOR_BGRA2GRAY)
                cv2.imshow('gray1', gray1)
                cv2.imshow('gray2', gray2)
                cv2.waitKey()
                # Detecting Chess board corners
                success1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
                success2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
                print(success1, success2)

                if success1 and success2:
                    successes += 1

                    corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                    corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

                    # Saving the imgpoints and the objpoints
                    objpoints.append(objp)
                    imgpoints1.append(corners1)
                    imgpoints2.append(corners2)

                    # Draw and display the corners
                    cv2.drawChessboardCorners(image1, chessboard_size, corners1, success1)
                    cv2.drawChessboardCorners(image2, chessboard_size, corners2, success2)
                    cv2.imshow('Image 1', image1)
                    cv2.imshow("Image 2", image2)
                    cv2.waitKey(1500)

                cv2.destroyAllWindows()

            # Break if enough images for calibration are collected.
            if successes == n_images:
                # Calibration
                ret1, cameraMatrix1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, frameSize, None, None)
                ret2, cameraMatrix2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, frameSize, None, None)

                h1, w1, ch1 = image1.shape
                h2, w2, ch2 = image2.shape

                newCameraMatrix1, roi_1 = cv2.getOptimalNewCameraMatrix(cameraMatrix1, dist1, (w1, h1), 1, (w1, h1))
                newCameraMatrix2, roi_2 = cv2.getOptimalNewCameraMatrix(cameraMatrix2, dist2, (w2, h2), 1, (w2, h2))

                # Stereo Vision Calibration
                flags = 0
                flags |= cv2.CALIB_FIX_INTRINSIC

                criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                retStereo, newCameraMatrix1, dist1, newCameraMatrix2, dist2, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints,
                                                                                                                                                  imgpoints1,
                                                                                                                                                  imgpoints2,
                                                                                                                                                  newCameraMatrix1,
                                                                                                                                                  dist1,
                                                                                                                                                  newCameraMatrix2,
                                                                                                                                                  dist2,
                                                                                                                                                  gray1.shape[::-1],
                                                                                                                                                  criteria_stereo,
                                                                                                                                                  flags)

                rectifyScale = 1
                rect1, rect2, projMatrix1, projMatrix2, Q, roi1, roi2 = cv2.stereoRectify(newCameraMatrix1,
                                                                                          dist1,
                                                                                          newCameraMatrix2,
                                                                                          dist2,
                                                                                          gray1.shape[::-1],
                                                                                          rot,
                                                                                          trans,
                                                                                          rectifyScale,
                                                                                          (0, 0))
                stereoMap1 = cv2.initUndistortRectifyMap(newCameraMatrix1, dist1, rect1, projMatrix1, gray1.shape[::-1],
                                                         cv2.CV_16SC2)
                stereoMap2 = cv2.initUndistortRectifyMap(newCameraMatrix2, dist2, rect2, projMatrix2, gray2.shape[::-1],
                                                         cv2.CV_16SC2)

                print("Saving parameters!")
                cv_file = cv2.FileStorage(f'{name}stereoMap.xml', cv2.FILE_STORAGE_WRITE)
                cv_file.write('stereoMap1_x', stereoMap1[0])
                cv_file.write('stereoMap1_y', stereoMap1[1])
                cv_file.write('stereoMap2_x', stereoMap2[0])
                cv_file.write('stereoMap2_y', stereoMap2[1])

                cv_file.release()
                break



    def triangulate_face(self, img1, img2):
        """

        :param img1 & img2:  Rectified images from both cameras
        :return:
        """
        faces1 = self.face_detection(img1)
        faces2 = self.face_detection(img2)
        # face = center point of face
        face1 = self.handle_faces(faces1)
        face2 = self.handle_faces(faces2)
        if len(face2) == 0 or len(face1) == 0:
            # one camera could not detect the face
            return face1, face2
        ##### Calculating Depth ######
        h1, w1, c1 = img1.shape
        h2, w2, c2 = img2.shape

        if w1 == w2:
            f_pixel = (w1*0.5) / np.tan(self.alpha * 0.5 * np.pi/180)
        else:
            print('Left and right camera do not have the same resolution! Fix in a preprocessing step!')

        x1 = face1[0]
        x2 = face2[0]

        return face1, face2


class spatialFacePosition(faceTriangulation):
    """
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
    def __init__(self, monitor_center: list,
                 monitor_normal: list,
                 monitor_rotation: list,
                 face_detection=faceDetection(),
                 cap1 = cv2.VideoCapture(0,  cv2.CAP_DSHOW),
                 cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW),
                 handle_faces = faceHandeler(),
                 frame_rate = 30,
                 B = 11.5,
                 f = 3.8,
                 alpha = 120,
                 name = ''):
        """
        Aside from attributing the monitor center vector and the monitor normal vector to the class instance, also
        the

        :param monitor_center:        [list] Vector pointing from the "main"-camera towards the monitor center. This
                                      vector is based on the main-cameras coordinate system.
        :param monitor_normal:        [list] Plane-normal of the monitor, based on the "main"-cameras coordinate system.
                                      This vector is based on the main-cameras coordinate system.
                                      It will also function as the x-axis in the screen coordinate system.
        :param monitor_rotation:      [list] Vector that is parallel to the lower edge of the screen. It is also based
                                      on the main-cameras coordinate system and will function as the z-coordinate in
                                      the screen coordinate system.
        :param face_detection:        [callable Object] Function or callable class that takes in an image and returns
                                      the coordinates of the face or position between the eyes.
        :param cap1 u. cap2:          cv2 video-streams from both cameras.
        :param handle_faces:          Callable object that chooses the right face from the detected list of faces.
        :param frame_rate:            Frame rate of both cameras.
        :param B:                     Distance between both cameras [cm].
        :param f:                     Focal length of the cameras [mm].
        :param alpha:                 Horizontal field of view [degrees].
        :param name:                  Name of this configuration.

                                       
        """
        super().__init__()
        self.face_detection = face_detection
        self.cap1 = cap1
        self.cap2 = cap2
        self.handle_faces = handle_faces
        self.frame_rate = frame_rate
        self.B = B
        self.f = f
        self.alpha = alpha
        self.name = name

        self.monitor_center = monitor_center
        self.monitor_normal = monitor_normal
        self.monitor_rotation = monitor_rotation

        y = np.cross(self.monitor_rotation, self.monitor_normal)

        # Axes of the screen coordinate system.
        self.new_x = normalize(monitor_normal)
        self.new_y = normalize(y)
        self.new_z = normalize(monitor_rotation)

        ######################## Loading Initial Camera Calibration ##########################
        try:
            cv_file = cv2.FileStorage()
            cv_file.open(f'{self.name}stereoMap.xml', cv2.FileStorage_READ)
        except FileNotFoundError:
            # Calibrating to generate a stereMap named {name}stereoMap.xml
            self.calibrate(self.name)
            cv_file = cv2.FileStorage()
            cv_file.open(f'{self.name}stereoMap.xml', cv2.FileStorage_READ)

        self.stereoMap1_x = cv_file.getNode('stereoMap1_x').mat()
        self.stereoMap1_y = cv_file.getNode('stereoMap1_y').mat()
        self.stereoMap2_x = cv_file.getNode('stereoMap2_x').mat()
        self.stereoMap2_y = cv_file.getNode('stereoMap2_y').mat()


    def __call__(self):
        """
        Reads both streams, loads just one frame and predicts the spatial position of the head in relation to the
        screen.

        :return:    Success face_pos img1 (with face annotations) img2 (with face annotations)
        """

        sucess1, img1 = self.cap1.read()
        sucess2, img2 = self.cap2.read()

        if not sucess2 or not sucess1:
            print('Could not read both Cams')
            return False

        ##### Frame Rectification #####
        # Undisort and rectify images using the stereoMap generated during calibration

        img1 = cv2.remap(img1, self.stereoMap1_x, self.stereoMap1_y, cv2.INTER_LANCZOS4)
        img2 = cv2.remap(img2, self.stereoMap2_x, self.stereoMap2_y, cv2.INTER_LANCZOS4)

        face1, face2 = self.triangulate_face(img1, img2)

        return face1, face2, img1, img2



# Function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        print("Left mouse button pressed! Capturing image.")
        cv2.imwrite('captured_image.jpg', img1)
        print("Image captured and saved as 'captured_image.jpg'")


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


    faceposition = spatialFacePosition([0, 0, 0], [1, 0, 0], [0, 0, 1])

    for i in range(10000):
        face1, face2, img1, img2 = faceposition()
        try:
            cv2.circle(img1, face1[0].astype(np.int32), 5, (255, 0, 255), cv2.FILLED)
        except Exception:
            pass
        try:
            cv2.circle(img2, face2[0].astype(np.int32), 5, (255, 0, 255), cv2.FILLED)
        except Exception:
            pass

        cv2.imshow(f'Image 1', img1)
        cv2.imshow(f'Image 2', img2)
        cv2.waitKey(1)

    #faceposition.calibrate()

    #calibrate_stereo_camera(cap1, cap2)
    #cv2.namedWindow('Webcam')
    #cv2.setMouseCallback('Webcam', mouse_callback)
    #sucess1, img1old = cap1.read()
    #sucess2, img2old = cap2.read()
    #img2old = np.array(img2old, dtype=np.float64)
    #previous_images = [img2old]
    #set_back = 20
    #while cap1.isOpened():
    #    sucess1, img1 = cap1.read()
    #    sucess2, img2 = cap2.read()
    #    img2copy = np.array(img2, dtype=np.float64)

    #    img2 = np.array(img2, dtype=np.float64)
    #    img2 = img2 - 20 - img2old
#
    #    img2[img2 < 0] = 0

        #img2 = cv2.imread("captured_image.jpg")
        #img2 = cv2.resize(img2, [1920, 1080])
        ####
        #img1old += img1
     #   img2old = img2copy

     #   cv2.imshow('Webcam', img2.astype(np.uint8))
     #   k = cv2.waitKey(1)
     #   if not k == -1: break
        #img1, faces1 = detector.findFaceMesh(img1, draw=True)
        #img2, faces1 = detector.findFaceMesh(img2, draw=True)
        #cv2.imshow('Camera 1', img1)
        #cv2.imshow('Camera 2', img2)

        #cv2.waitKey(1)
