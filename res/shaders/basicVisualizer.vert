#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNom;

out vec3 uNom;

uniform mat4 transformation;

void main()
{
   gl_Position = transformation * vec4(aPos, 1.0);
   //vec4 uN = transformation * vec4(aNom, 1.0f);
   uNom = aNom;//vec3(uN.x, uN.y, uN.z);
}