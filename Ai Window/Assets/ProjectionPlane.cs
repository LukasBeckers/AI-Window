using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ProjectionPlane : MonoBehaviour
{
    public float height;
    public float width;
    public bool drawGizmo;

    


    // Start is called before the first frame update
    void Start()
    {
        height = 29.5f;
        width = 52.5f;

        drawGizmo = true;


    }

    // Update is called once per frame
    void Update()
    {
        Vector3 bottomLeft = transform.TransformPoint(new Vector3(-width, -height) * 0.5f);
        Vector3 bottomRight = transform.TransformPoint(new Vector3(width, -height) * 0.5f);
        Vector3 topLeft = transform.TransformPoint(new Vector3(-width, height) * 0.5f);
        Vector3 topRight = transform.TransformPoint(new Vector3(width, height) * 0.5f);

        Vector3 dirRight = (bottomRight - bottomLeft).normalized;
        Vector3 dirUp = (topLeft - bottomLeft).normalized;
        Vector3 dirNormal = -Vector3.Cross(dirRight, dirUp).normalized;

        Matrix4x4 mbox = Matrix4x4.zero;
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

        Debug.Log("Update called!");
        
        }
    void OnDrawGizmo ()
    {
       
    }
    }
