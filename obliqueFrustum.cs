using UnityEngine;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System;

public class LookAtObliqueFrustum : MonoBehaviour
{
    // For the Socket
    private TcpClient tcpClient;
    private NetworkStream networkStream;
    private Thread receiveThread;

    // Screen and Camera Parameters
    public Vector3 screenCenter;
    public Vector3 cameraStart;
    public float width;
    public float height;
    public float moveSpeed;
    private Camera cam;
    private GameObject screen;

    // For CalculateScreenCorners function
    private Vector3 lowerLeft;
    private Vector3 lowerRight;
    private Vector3 upperLeft;
    private Vector3 upperRight;
    private Vector3 dirRight;
    private Vector3 dirUp;
    private Vector3 dirNormal;
    private Matrix4x4 mbox;

    // For OffCenterProjection
    private Vector3 va;
    private Vector3 vb;
    private Vector3 vc;
    private Vector3 vd;
    private Vector3 eyePos;
    private Vector3 viewDir;
    private float d;
    private float near;
    private float far;
    private float nearOverDist;
    private float left;
    private float right;
    private float lower;
    private float upper;
    private Matrix4x4 projectionMatrix;
    private Matrix4x4 R;
    private Matrix4x4 T;

    // For RotateScreen
    private bool rotationMode;
    public float rotationSpeed;

    // for the position calculations
    private List<float> last_xpositions;
    private List<float> last_ypositions;
    private List<float> last_zpositions;
    private int xsmooting;
    private int ysmooting;
    private int zsmooting;

    void Start()
    {
        // Connecting the Socket
        ConnectToServer();

        // !!! here the default values are definde
        this.screenCenter = new Vector3(30, 15, 30);
        this.height = 0.295f ;
        this.width = 0.525f ;
        this.mbox = Matrix4x4.zero;
        this.rotationMode = false;
        this.last_xpositions = new List<float>();
        this.last_ypositions = new List<float>();
        this.last_zpositions = new List<float>();
        this.moveSpeed = 5f;
        this.xsmooting = 1;
        this.ysmooting = 1;
        this.zsmooting = 1;
        this.rotationSpeed = 50.0f;

        // Calculate the initial screen corners
        CreateScreenCorners();      

        // Move the camera to the start position
        cam = GetComponent<Camera>();
        cam.transform.position = this.screenCenter + this.dirNormal; 
    }

    void CreateScreenCorners()
    {
        // Creates Screen Corners
        // These Corners are not rotated initially, this will be done by second function.
        // Initially it will be assumed, that the target point is the center of the screen.
        // The width will be represented in the x direction and the hight in the y direction.          
        this.lowerLeft = this.screenCenter + new Vector3(-this.width * 0.5f, -this.height * 0.5f, 0);
        this.lowerRight = this.screenCenter + new Vector3(this.width * 0.5f, -this.height * 0.5f, 0);
        this.upperLeft = this.screenCenter + new Vector3(-this.width * 0.5f, this.height * 0.5f, 0);
        this.upperRight = this.screenCenter + new Vector3(this.width * 0.5f, this.height * 0.5f, 0);

        this.dirRight = (this.lowerRight - this.lowerLeft).normalized;
        this.dirUp = (this.upperLeft - this.lowerLeft).normalized;
        this.dirNormal = -Vector3.Cross(this.dirRight, this.dirUp).normalized;

        
        mbox[0, 0] = dirRight.x;
        mbox[0, 1] = dirRight.y;
        mbox[0, 2] = dirRight.z;

        mbox[1, 0] = dirUp.x;
        mbox[1, 1] = dirUp.y;
        mbox[1, 2] = dirUp.z;

        mbox[2, 0] = dirNormal.x;
        mbox[2, 1] = dirNormal.y;
        mbox[2, 2] = dirNormal.z;

        mbox[3, 3] = 1.0f;
     }

    void RotateScreen()
    {
        // Can be used to rotate the screen plane around the screen center.
        // Mouse should be used to rotate screen plane.
        // W and S should be used to move the screen forward and backward.
        // A and D should be used to move the screen left and right, space and shift for up and down.

        float horizontalInput = Input.GetKey(KeyCode.A) ? 1 : (Input.GetKey(KeyCode.D) ? -1 : 0);
        Vector3 horizontal = horizontalInput * moveSpeed * Time.deltaTime * -dirRight;
        float verticalInput = Input.GetKey(KeyCode.W) ? 1 : (Input.GetKey(KeyCode.S) ? -1 : 0);
        Vector3 vertical = verticalInput * moveSpeed * Time.deltaTime * -dirNormal;
        float updownInput = Input.GetKey(KeyCode.Space) ? 1 : (Input.GetKey(KeyCode.LeftShift) ? -1 : 0);
        Vector3 updown = updownInput * moveSpeed * Time.deltaTime * dirUp;

        // Calculate the new camera position after translation
        Vector3 newCameraPosition = transform.position + horizontal + updown + vertical;
        

        // Translate targetPoint and screen corners.
        this.screenCenter = this.screenCenter + horizontal + updown + vertical;
        this.lowerLeft = this.lowerLeft + horizontal + updown + vertical;
        this.lowerRight = this.lowerRight + horizontal + updown + vertical;
        this.upperLeft = this.upperLeft + horizontal + updown + vertical;
        this.upperRight = this.upperRight + horizontal + updown + vertical;

        // Rotate the screen corners around the target poins
        float rotationAmmountX = Input.GetAxis("Mouse X") * Time.deltaTime;
        float rotationAmmountY = Input.GetAxis("Mouse Y") * Time.deltaTime;

        

        List<Vector3> screenCorners = new List<Vector3> {this.lowerLeft, this.lowerRight, this.upperLeft, this.upperRight, newCameraPosition};

        for (int i = 0; i < screenCorners.Count; i++)
        {
            Vector3 relativePosition = screenCorners[i] - this.screenCenter;

            Quaternion rotationX = Quaternion.Euler(0, rotationAmmountX * rotationSpeed, 0);
            Quaternion rotationY = Quaternion.Euler(-rotationAmmountY * rotationSpeed, 0, 0);
            screenCorners[i] = this.screenCenter + rotationX  * rotationY * relativePosition;
        }
        this.lowerLeft = screenCorners[0];
        this.lowerRight = screenCorners[1];
        this.upperLeft = screenCorners[2];
        this.upperRight = screenCorners[3];
        newCameraPosition = screenCorners[4];

        // Recalculate mbox for the rotated screen corners.
        this.dirRight = (this.lowerRight - this.lowerLeft).normalized;
        this.dirUp = (this.upperLeft - this.lowerLeft).normalized;
        this.dirNormal = -Vector3.Cross(this.dirRight, this.dirUp).normalized;


        mbox[0, 0] = dirRight.x;
        mbox[0, 1] = dirRight.y;
        mbox[0, 2] = dirRight.z;

        mbox[1, 0] = dirUp.x;
        mbox[1, 1] = dirUp.y;
        mbox[1, 2] = dirUp.z;

        mbox[2, 0] = dirNormal.x;
        mbox[2, 1] = dirNormal.y;
        mbox[2, 2] = dirNormal.z;

        mbox[3, 3] = 1.0f;

        // Move the camera
        cam.transform.position = newCameraPosition;
        
        Debug.Log("Rotation Ammount x" + rotationAmmountX);
        Debug.Log("" + newCameraPosition);
    }

    private void ConnectToServer()
    {
        tcpClient = new TcpClient("127.0.0.1", 12345); // Server IP and Port
        networkStream = tcpClient.GetStream();

        // Start a thread to receive data continuously
        receiveThread = new Thread(new ThreadStart(ReceiveDataThread));
        receiveThread.Start();
    }

    private string ReceiveData()
    {
        byte[] buffer = new byte[4096];
        int bytesRead = networkStream.Read(buffer, 0, buffer.Length);
        return Encoding.UTF8.GetString(buffer, 0, bytesRead);
    }

    private void ReceiveDataThread()
    {
        while (true)
        {
            // do nothing
        }
    }

    private void OnApplicationQuit()
    {
        // Clean up resources when the application is closed
        if (tcpClient != null)
        {
            tcpClient.Close();
        }
    }

    void CheckPosition(Vector3 newPosition)
    {
        // Should be called in the LateUpdate method bevore the frustum is calculated. 
        // This function should retrun false if the newPosition, the camera is about to occupy is not allowed ( like a position behind the screen plane)

    }

    Vector3 SmoothPosition(Vector3 Position)
    {
        // Should be called in LateUpdate before the camera-position is updated.

        this.last_xpositions.Add(Position.x);
        this.last_ypositions.Add(Position.y);
        this.last_zpositions.Add(Position.z);
        Vector3 SmoothedPosition = Vector3.zero;

        foreach (var val in this.last_xpositions)
        {
            SmoothedPosition.x += val;
        }

        SmoothedPosition.x /= last_xpositions.Count;

        foreach (var val in this.last_ypositions)
        {
            SmoothedPosition.y += val;
        }

        SmoothedPosition.y /= last_ypositions.Count;

        foreach (var val in this.last_zpositions)
        {
            SmoothedPosition.z += val;
        }

        SmoothedPosition.z /= last_zpositions.Count;

        // remove oldest position if last_positions is too long
        if (last_xpositions.Count > xsmooting) { last_xpositions.RemoveAt(0); }
        if (last_ypositions.Count > ysmooting) { last_ypositions.RemoveAt(0); }
        if (last_zpositions.Count > zsmooting) { last_zpositions.RemoveAt(0); }

        return SmoothedPosition;

    }

    void OffCenterProjection()
    {
        // This function calcuates the projection and the view matrix (worldToCameraMatrix)
        // This function will be called in the LateUpdate method.

        eyePos = transform.position;

        va = this.lowerLeft - eyePos;
        vb = this.lowerRight - eyePos;
        vc = this.upperLeft - eyePos;
        vd = this.upperRight - eyePos;

        viewDir = eyePos + va + vb + vc + vd;
        d = -Vector3.Dot(va, this.dirNormal);
        near = d;
        far = cam.farClipPlane;
        cam.nearClipPlane = 0.05f;
        

        nearOverDist = near / d;
        left = Vector3.Dot(this.dirRight, va) * nearOverDist;
        right = Vector3.Dot(this.dirRight, vb) * nearOverDist;
        lower = Vector3.Dot(this.dirUp, va) * nearOverDist;
        upper = Vector3.Dot(this.dirUp, vc) * nearOverDist;
        
        projectionMatrix = Matrix4x4.Frustum(left, right, lower, upper, near, far);
        T = Matrix4x4.Translate(-eyePos);
        R = Matrix4x4.Rotate(Quaternion.Inverse(transform.rotation) * transform.rotation);

        cam.worldToCameraMatrix = mbox * R * T;
        cam.projectionMatrix = projectionMatrix;

    }


    private Vector3 TransformToUnityCoordinates(Vector3 position)
    {
        
        // Create a transformation matrix
        Matrix4x4 transformationMatrix = new Matrix4x4();
        transformationMatrix.SetColumn(0, new Vector4(this.dirNormal.x, this.dirNormal.y, this.dirNormal.z, 0));
        transformationMatrix.SetColumn(1, new Vector4(this.dirUp.x, this.dirUp.y, this.dirUp.z, 0));
        transformationMatrix.SetColumn(2, new Vector4(this.dirRight.x, this.dirRight.y, this.dirRight.z, 0));
        transformationMatrix.SetColumn(3, new Vector4(this.screenCenter.x, this.screenCenter.y, this.screenCenter.z, 1));

        // Apply the transformation matrix to position
        Vector4 position_homogeneous = new Vector4(position.x, position.y, position.z, 1);
        Vector4 unityCoordinates_homogeneous = transformationMatrix.MultiplyPoint(position);
        Vector3 unityCoordinates = new Vector3(unityCoordinates_homogeneous.x, unityCoordinates_homogeneous.y, unityCoordinates_homogeneous.z);
        return unityCoordinates;
    }


    void LateUpdate()
    {   
        if (!rotationMode)
        {       
            // Checking if rotationMode should be activated
            if (Input.GetKeyDown(KeyCode.R)) { rotationMode = true; }

            // Get input from arrow keys for movement
            //Vector3 horizontal = Input.GetKey(KeyCode.W) ? 1 : (Input.GetKey(KeyCode.S) ? -1 : 0) * moveSpeed * Time.deltaTime * dirRight;
            //float horizontalInput = Input.GetKey(KeyCode.A) ? 1 : (Input.GetKey(KeyCode.D) ? -1 : 0);
            //Vector3 horizontal = horizontalInput * moveSpeed * Time.deltaTime * -dirRight;
            //float verticalInput = Input.GetKey(KeyCode.W) ? 1 : (Input.GetKey(KeyCode.S) ? -1 : 0);
            //Vector3 vertical = verticalInput * moveSpeed * Time.deltaTime * -dirNormal;
            //float updownInput = Input.GetKey(KeyCode.Space) ? 1 : (Input.GetKey(KeyCode.LeftShift) ? -1 : 0);
            //Vector3 updown = updownInput * moveSpeed * Time.deltaTime * dirUp;

            //Vector3 vertical = Input.GetAxis("Vertical") * moveSpeed * Time.deltaTime * -dirNormal;
            //Vector3 updown = Input.GetAxis("UpDown") * moveSpeed * Time.deltaTime * -dirNormal;
            //Vector3 newPosition = transform.position + horizontal + updown + vertical;


            // Move the camera
            //cam.transform.position = newPosition;

            // Listen to the socket and move based on the data recived from the socket
            if (networkStream != null && networkStream.DataAvailable)
            {
                
                    string receivedData = ReceiveData();
                    // Process the received JSON data
                    // Position coordinates are in relation to the screen center
                    Vector3 rawCoordinates = JsonUtility.FromJson<Vector3>(receivedData);
                    Vector3 position = this.TransformToUnityCoordinates(rawCoordinates);
                    Vector3 smoothedPosition = this.SmoothPosition(position);
                    Debug.Log("Display Center: " + this.screenCenter);
                    Debug.Log("Received coordinates: " + smoothedPosition);
                    Debug.Log("Raw Coordinates" + rawCoordinates);

                    cam.transform.position = smoothedPosition;
                }
                
          
        }
        else
        {   
            // Checking if rotationMode should be deactivated. (R Key to togg
            if (Input.GetKeyDown(KeyCode.R)) { rotationMode = false; }
            // rotation mode
            RotateScreen();
        }

        OffCenterProjection();
    }
}

public class Vector3Wrapper
{
    public Vector3 value;

    public Vector3Wrapper(Vector3 initialValue)
    {
        value = initialValue;
    }
}

