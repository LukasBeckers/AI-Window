// Stencil shader to mark the area
Shader "Custom/StencilMaskShader"
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
                Comp always
                Pass replace
            }

            // Rest of the shader pass
        }
    }
}
