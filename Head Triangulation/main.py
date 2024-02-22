from face_triangulation import *
import socket
import json
import time


if __name__ == "__main__":
    while True:
        try:
            # Creating the spatial face position object
            spatial_face_position = spatialFacePosition("Setup Aachen 1", face_detection=faceDetection())

            # Configuring the socket
            HOST = '127.0.0.1'  # Server IP address (localhost in this example)
            PORT = 12345  # Port to bind to

            # Create a socket object
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Bind the socket to a specific address and port
            server_socket.bind((HOST, PORT))

            # Listen for incoming connections (1 connection at a time)
            server_socket.listen(1)
            print(f"Server listening on {HOST}:{PORT}")

            # Accept a connection from the client
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address}")

            try:
                while True:
                    start_time = time.time()
                    coordinates, _, _, _, _ = spatial_face_position()
                    coordinates = {'x': coordinates[0], 'y': coordinates[1], 'z': coordinates[2]}

                    # Convert the coordinates to a JSON string
                    coordinates_json = json.dumps(coordinates)

                    print(f"Sending coordinates {coordinates}")

                    # Send the JSON string to the client
                    client_socket.sendall(coordinates_json.encode())
                    print(f"Update took {time.time() - start_time}s")

            finally:
                # Clean up the connection
                client_socket.close()
                server_socket.close()
        except:
            pass