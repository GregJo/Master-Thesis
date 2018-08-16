#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

//Errors to cover:
class GraphicsWindow
{
public:

	GraphicsWindow() {}
	GraphicsWindow(int width, int height, const char* name, GLFWmonitor* monitor, GLFWwindow* share, GLFWkeyfun cbfun);
	GraphicsWindow(int width, int height, const char* name, GLFWmonitor* monitor, GLFWwindow* share, GLFWkeyfun cbfun, std::vector<int> targets, std::vector<int> hints);
	~GraphicsWindow();

	void release();

	GLFWwindow* getWindowHandle();
	
	int getWindowHeight();
	int getWindowWidth();

	//! \brief Checking if the window should close flag should be set.
	//! \return Returns the int should close flag.
	int shouldClose();
	
	//! \brief Make window current context.
	void makeContextCurrent();
	
	void swapBuffers();

private:

	GLFWwindow* m_windowHandle;
	int m_height;
	int m_width; 
};

static void error_callback(int error, const char* description)
{
    fputs(description, stderr);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

static void setErrorCallbackAndInit(GLFWerrorfun cbfun)
{
	glfwSetErrorCallback(cbfun);

	if (!glfwInit())
		exit(1);
}

static void initGlew()
{
	// Initialize GLEW
	GLenum glewError = glewInit() ;
	if ( glewError != GLEW_OK )  {
		// Problem : glewInit failed, something is seriously wrong.
		printf( "Failed to initialize GLEW ! Error : %s\n", glewGetErrorString( glewError ) ) ;
		glfwTerminate() ;
		exit(1) ;
	}
}