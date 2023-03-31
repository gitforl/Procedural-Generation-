#version 330 core

in vec3 uCol;

out vec4 FragColor;

void main()
{
    vec3 tempCol = vec3(1.0f, 0.0f, 0.0f); 
    FragColor = vec4(uCol, 1.0f);
}