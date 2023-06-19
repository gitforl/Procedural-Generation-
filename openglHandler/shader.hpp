#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <string>

class Shader{
    private:
        unsigned int shaderProgram;

        void InitializeShader(std::string const &filename)
        {

            unsigned int shader;
            std::ifstream fileSrc(filename);

            auto filenameDotIndex = filename.rfind(".");
            auto fileType = filename.substr(filenameDotIndex + 1); 

            if(fileSrc.fail())
            {
                std::cout << "Failed Loading Shader Script: " << std::endl;
                return;
            }

            auto src = std::string(std::istreambuf_iterator<char>(fileSrc),
                                  (std::istreambuf_iterator<char>()));

            const char * source = src.c_str();

            const char * ShaderTypeCaps;
    
            if(fileType == "vert"){
                shader = glCreateShader(GL_VERTEX_SHADER);
                ShaderTypeCaps = "VERTEX";
            }
            if(fileType == "frag")   
                shader = glCreateShader(GL_FRAGMENT_SHADER);
                ShaderTypeCaps = "FRAGMENT";
            glShaderSource(shader, 1, &source, NULL);
            glCompileShader(shader);

            int success;
            char infoLog[512];
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if(!success)
            {
                glGetShaderInfoLog(shader, 512, NULL, infoLog);
                std::cout << "ERROR::SHADER::" << ShaderTypeCaps << "::COMPILATION_FAILED\n" << infoLog << std::endl;
            }

            glAttachShader(shaderProgram, shader);
            glDeleteShader(shader);
        };

        void LinkShaders()
        {
            glLinkProgram(shaderProgram);

            int success;
            char infoLog[512];

            glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
            if(!success)
            {
                glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
                std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
            }
        };
    public:
        Shader(){};

        ~Shader(){};

        void Use()
        {
            glUseProgram(shaderProgram);
        };

        unsigned int GetId()
        {
            return shaderProgram;
        };


        void Create(std::string const &vertPath, std::string const &fragPath)
        {
            shaderProgram = glCreateProgram();
            InitializeShader(vertPath);
            InitializeShader(fragPath);
            LinkShaders();
        };  
};
