#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aCol;

out vec3 uCol;

uniform mat4 transformation;

void main()
{
   gl_Position = transformation * vec4(aPos, 1.0);
   uCol = aCol;
}