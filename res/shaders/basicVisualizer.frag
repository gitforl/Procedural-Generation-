#version 330 core

in vec3 uNom;

out vec4 FragColor;

void main()
{
   vec3 ambient = 0.1f * vec3(1.0f, 1.0f, 1.0f);
   vec3 lightDir = vec3(0.0f, 1.0f, 0.0f);
   vec3 lightColor = vec3(1.0f, 1.0f, 1.0f);


   vec3 diffuse = (0.25f + 0.5 * max(dot(normalize(uNom),lightDir), 0.0f)) * lightColor;
   vec3 colorValues = ambient + diffuse;
   FragColor = vec4(colorValues, 1.0f);
}