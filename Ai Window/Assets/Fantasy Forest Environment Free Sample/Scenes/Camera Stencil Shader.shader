// Camera shader to render only the stenciled area
Shader "Custom/CameraStencilShader"
{
    Properties
    {
        // Define shader properties here if needed
    }
    SubShader
    {
        Tags { "Queue"="Geometry" "RenderType"="Opaque" }
        Pass
        {
            Stencil
            {
                Ref 1
                Comp equal
            }

            // Rest of the shader pass
        }
    }
}