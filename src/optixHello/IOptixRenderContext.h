#pragma once

#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

/*
This renderer has in first place to be conform with monte carlo based rendering (path tracing), mostly real time/online rendering, 
which means it should also be possible to render a single image/offline only
This abstraction should not yet include the adaptive rendering extension. I will build the adaptive rendering on top of this.

Functionality:
	- setup context
	- setup scene
		- setup camera
		- setup geometry
	- setup display window
	- setup keyboard/mouse input
	- display
	- clean up

For now i will be using glut, might switch to glfw.
Also i will need a logger, i plan to use the long bow logger.
*/
using namespace optix;

class BasicLight;

class IOptixRenderContext 
{
public:

	virtual ~IOptixRenderContext() {}

	virtual void createContext(const char* const SAMPLE_NAME, const char* fileNameOptixPrograms, const bool use_pbo, int rr_begin_depth) {}
	//virtual void setupScene(float3 camera_eye, float3 camera_up, float3 camera_lookat, BasicLight* lights, const unsigned int numberOfLights, const std::string& filename) = 0;
	
	// TODO: Find a better solution, more general for cameras, instead of just using predeclare (of TrackballCamera):
	virtual void display(class TrackballCamera* camera) {}
	//virtual void setupDisplayWindow() = 0;
	//virtual void setupInput() = 0;
	virtual void clean() {}

protected:

	optix::Context m_context;
	uint32_t       m_width;
	uint32_t       m_height;
};