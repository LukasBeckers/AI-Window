from spatial_face_position import *
import pickle as pk


def configure_spatial_face_positon(name,
                                   cameras):
        """
        stores the configuration of a spatialFacePosition setup.
        :param name:            Name of this setup, using this name the configuration will be stored in the
                                face_position_configuration.json file.
        :param cameras:         Name of all cameras that will be used in this setup, the cameras must have a config
                                using this name in the camera_configurations.json.
        """
        # all configurations are stored in a single json file separated by their setup-names
        try:
            with open('face_position_configurations.json', 'r') as json_file:
                configurations = json.load(json_file)
        except FileNotFoundError:
            configurations = {}

        if name in configurations.keys():
            print(f"Spatial face detection configurations named {name} already exists, aborting!")
            return

            # storing the general setup configurations
        configurations[name] = {
            "cameras": cameras,
        }

        # saving the dict as JSON
        with open('face_position_configurations.json', 'w') as json_file:
            json.dump(configurations, json_file)


def configure_camera(name, stream, resolution=[480, 640]):
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
            print('Successfully loaded previously written camera configurations')
    except FileNotFoundError:
        print('Could not load previous camera configurations file, creating empty file.')
        configurations = {}

    if name in configurations.keys():
        print(f"Camera configurations named {name} already exists, aborting!")
        return

    # storing the stream of the camera in a JSON format.
    configurations[name] = {
        "stream": stream,
        "resolution": resolution  # height, width
    }

    # saving the dict as JSON
    with open('camera_configuration.json', 'w') as json_file:
        print(configurations)
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

    configure_camera("Camera0_1080", stream=0, resolution=[1080, 1920])
    configure_camera("Camera1_1080", stream=1, resolution=[1080, 1920])
    configure_spatial_face_positon("Setup Aachen 1 1080", ["Camera0_1080", "Camera1_1080"])
    spatial_face_position = spatialFacePosition("Setup Aachen 1 1080", face_detection=faceDetection())

    for i, camera in enumerate(spatial_face_position.cameras[1:]):
        continue
        print("Fisheye Calibrating camera", camera, camera.fisheye, camera.resolution)
        camera.fisheye_calibrate(rows=7, columns=9, n_images=10, scaling=0.025)
        print("Normal Calibration")
        camera.calibrate(scaling=0.025, n_images=10, rows=7, columns=9, fisheye=True)

    #spatial_face_position.stero_calibrate(scaling=0.025, n_images=5, rows=7, columns=9)

    #spatial_face_position.triangulate_checkerboard( )
    #spatial_face_position.calibrate_display_center()

    if True:   # Displays the triangulation results.
        face_coordinates = []
        frames = []
        while True:
            coordinate, frame1, frame2, face1, face2 = spatial_face_position()

            face_coordinates.append(coordinate)

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
        cam1_pos = spatial_face_position.transform_point([0, 0, 0], spatial_face_position.transform_matrix)
        ax.scatter(cam1_pos[0], cam1_pos[1], cam1_pos[2], c="green", s=15)  # Adjust the size (s) to make it larger

        # Camera 2
        cam2_pos = [-x[0] for x in spatial_face_position.translation_matrix]
        cam2_pos = spatial_face_position.transform_point(cam2_pos, spatial_face_position.transform_matrix)
        ax.scatter(cam2_pos[0], cam2_pos[1], cam2_pos[2], c="green", s=15)
        face_coordinates = face_coordinates.tolist()

        # Plotting the display center
        ax.scatter(0,0,0, c="blue", s=50)

        face_coordinates.append([0,0,0])
        face_coordinates.append(cam1_pos)
        face_coordinates.append(cam2_pos)

        x_coords = [c[0] for c in face_coordinates]
        y_coords = [c[1] for c in face_coordinates]
        z_coords = [c[2] for c in face_coordinates]

        ax.set_xlim([min(x_coords), max(x_coords)])
        ax.set_ylim([min(y_coords), max(y_coords)])
        ax.set_zlim([min(z_coords), max(z_coords)])
        ax.set_box_aspect([np.ptp(x_coords), np.ptp(y_coords), np.ptp(z_coords)])
        plt.show()




