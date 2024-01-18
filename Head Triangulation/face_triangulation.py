import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

import numpy as np
import time



def calibrate_stereo_camera(stream_1, stream_2, name:str='unknown'):
    """
    Calibrates the cameras for stereovision via several chess-board images.
    The calibration function should be automated in the future by automatic chess-board detection

    :param stream_1: cv2.cap video stream of first camera
    :param stream_2: cv2.cap video stream of second camera
    :param name:     name of this setup (calibration parameters will be saved using this name)
    :return:         Calibration matrix?
    """
    calibration_images1 = []
    calibration_images2 = []
    chessboard_positions = ["Top Left", "Top Right", "Bottom Right", "Bottom Left", "Center"]

    while stream_1.isOpened() and stream_2.isOpened():
        success1, image1 = stream_1.read()
        success2, image2 = stream_2.read()

        cv2.putText(image1,
                    chessboard_positions[len(calibration_images1)],
                    (image1.shape[1]//2, image1.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2)

        cv2.putText(image2,
                    chessboard_positions[len(calibration_images2)],
                    (image2.shape[1]// 2, image2.shape[1]// 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2)



        cv2.imshow('Image1', image1)
        cv2.imshow('Image2', image2)

        k = cv2.waitKey(50)

        # Exit if esc is pressed
        if k == 27:
            break
        # Save Images if s is pressed
        elif k == ord('s'):
            calibration_images1.append(image1)
            calibration_images2.append(image2)

        # Break if enough images for calibration are collected.
        if len(calibration_images1) == len(chessboard_positions):
            break

    chessboard_size = (4,2)
    frame_size = calibration_images1[0].shape

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0)...
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0: chessboard_size[0], 0: chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] #3d points in real world space
    imgpoints1 = [] #2d points in image plane
    imgpoints2 = [] #2d points in image plane.

    for img1, img2 in zip(calibration_images1, calibration_images2):
        gray1, gray2 = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)

        # Detecting Chess board corners

        success1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
        success2, corners2 = cv2.findChessboardCorners(gray1, chessboard_size, None)
        print(success2, success1)
        if success1 and success2:

            objpoints.append(objp)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            imgpoints1.append(corners1)
            imgpoints2.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img1, chessboard_size, corners1, success1)
            cv2.drawChessboardCorners(img2, chessboard_size, corners2, success2)
            cv2.imshow('Image 1', img1)
            cv2.imshow("Image 2", img2)
            cv2.waitKey(1500)

    cv2.destroyAllWindows()


class faceTriangulation():
    """
    This is the main class used for face-triangulation.

    The spatial position of the face, or more optimally the middle point between the two eyes should be
    """
    def __init__(self):
        pass





if __name__ == "__main__":
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    calibrate_stereo_camera(cap1, cap2)
    #detector = FaceMeshDetector(maxFaces=2)


   # while cap.isOpened():
    #   start_time = time.time()
    #    sucess1, img1 = cap.read()
     #   sucess2, img2 = cap2.read()
     #   print(img1.shape)
        #img1, faces1 = detector.findFaceMesh(img1, draw=True)
        #img2, faces2 = detector.findFaceMesh(img2, draw=True)

     #   print('Time per frame', 1/(time.time() - start_time))
     #   print(len(faces1), len(faces2))

      #  cv2.imshow('Camera 1', img1)
       # cv2.imshow('Camera 2', img2)

       # cv2.waitKey(1)
