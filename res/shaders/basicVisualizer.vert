#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNom;

out vec3 uNom;

uniform mat4 model;
uniform mat4 view;

void main()
{
   gl_Position = view * vec4(aPos, 1.0);
   vec4 uN = view * vec4(aNom, 1.0f);
   uNom = vec3(uN.x, uN.y, uN.z);
}