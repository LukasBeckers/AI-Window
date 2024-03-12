using UnityEngine;

public class CameraLookAtAndMove : MonoBehaviour
{
    public Vector3 targetPoint = new Vector3(-10, 7, 70);
    public Vector3 cameraStart = new Vector3(-11, 7, 71);
    public Vector3 mainTargetPoint = new Vector3(100, 10, 100);
    public Vector3 maincameraStart = new Vector3(98, 10, 102);
    public float moveSpeed = 5.0f;
    public float rectangleHeight = 0.3f;
    public float rectangleWidth = 0.53f;
    public Camera secondaryCamera;

    void Start()
    {
        GameObject secondaryCameraObject = new GameObject("SecondaryCamera");
        secondaryCamera = secondaryCameraObject.AddComponent<Camera>();
        GameObject rectangle = GameObject.CreatePrimitive(PrimitiveType.Quad);
        rectangle.transform.position = targetPoint;
        rectangle.transform.localScale = new Vector3(rectangleWidth, rectangleHeight, 1);
        rectangle.transform.Rotate(0, 140, 0);
        secondaryCamera.transform.position = cameraStart;
       

        GetComponent<Camera>().depth = -1;
        GameObject mainRectangle = GameObject.CreatePrimitive(PrimitiveType.Quad);
        mainRectangle.transform.position = mainTargetPoint;
        mainRectangle.transform.localScale = new Vector3(rectangleWidth * 10, rectangleHeight * 10, 1);
        mainRectangle.transform.Rotate(0, 140, 0);
        transform.position = maincameraStart;
       
        
        RenderTexture renderTexture = new RenderTexture(1920, 1080, 32, RenderTextureFormat.ARGB32);
        secondaryCamera.targetTexture = renderTexture;

        Material material = new Material(Shader.Find("Standard"));
        material.mainTexture = renderTexture;
        mainRectangle.GetComponent<Renderer>().material = material;
    }

    void Update()
    {
        // Always look at the target point
        secondaryCamera.transform.LookAt(targetPoint);
        transform.LookAt(mainTargetPoint);

        // Get input from arrow keys for movement
        float horizontal = Input.GetAxis("Horizontal") * moveSpeed * Time.deltaTime;
        float vertical = Input.GetAxis("Vertical") * moveSpeed * Time.deltaTime;
        float updown = Input.GetAxis("UpDown") * moveSpeed * Time.deltaTime;

        // Move the camera
        secondaryCamera.transform.Translate(horizontal, updown, vertical);
        transform.Translate(-horizontal, -updown, vertical);
    }
}
