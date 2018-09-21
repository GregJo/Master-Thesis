#pragma once

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include "IOptixRenderContext.h"
#include "TrackballCamera.h"
#include <sutil.h>

class PathTraceRenderContext : public IOptixRenderContext
{

public:

	PathTraceRenderContext() { _context = 0; }
	virtual ~PathTraceRenderContext() { _destroyContext(); }

	void setWidth(uint32_t width) { _width = width; }
	void setHeight(uint32_t height) { _height = height; }

	void setFrameNumber(int frameNumber) { _frameNumber = frameNumber; }
	void setSqrtNumSample(int sqrt_num_samples) { _sqrt_num_samples = sqrt_num_samples; }

	void setNumSample(int num_samples) { _num_samples = num_samples; }

	void createContext(const char* const SAMPLE_NAME, const char* fileNameOptixPrograms, const bool use_pbo, int rr_begin_depth)
	{
		_context = Context::create();
		_context->setRayTypeCount(2);
		_context->setEntryPointCount(2);
		_context->setStackSize(1800);

		_context->setPrintEnabled(true);
		_context->setPrintBufferSize(2048);

		_context["scene_epsilon"]->setFloat(1.e-3f);
		_context["pathtrace_ray_type"]->setUint(0u);
		_context["pathtrace_shadow_ray_type"]->setUint(1u);
		_context["rr_begin_depth"]->setUint(rr_begin_depth);

		Buffer output_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["output_buffer"]->set(output_buffer);

		//createAndPassBufferFromTo("output_current_total_rays_buffer", "input_current_total_rays_buffer", RT_FORMAT_INT4, width, height, use_pbo);
		Buffer output_current_total_rays_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_INT4, _width, _height, use_pbo);
		_context["output_current_total_rays_buffer"]->set(output_current_total_rays_buffer);

		//passBufferFromTo("input_current_total_rays_buffer", "post_process_input_current_total_rays_buffer");

		////Buffer inputSceneRenderBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height);
		//Buffer inputBuffers =
		//	context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_BUFFER_ID, 1);
		//int* buffers = static_cast<int*>(inputBuffers->map());
		//buffers[0] = output_buffer->getId();
		//inputBuffers->unmap();
		//context["hoelder_adaptive_buffers"]->set(inputBuffers);

		// Setup programs
		const char *ptx = sutil::getPtxString(SAMPLE_NAME, fileNameOptixPrograms);
		_context->setRayGenerationProgram(0, _context->createProgramFromPTXString(ptx, "pathtrace_camera"));
		_context->setExceptionProgram(0, _context->createProgramFromPTXString(ptx, "exception"));
		_context->setMissProgram(0, _context->createProgramFromPTXString(ptx, "miss"));

		// Post processing
		Buffer output_scene_depth_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["output_scene_depth_buffer"]->set(output_scene_depth_buffer);

		//setupVarianceBuffer();

		// Adaptive source file
		//const char *adaptive_ptx = sutil::getPtxString(SAMPLE_NAME, "adaptiveOptixPathTracer.cu");

		//const char *variance_adaptive_ptx = sutil::getPtxString(SAMPLE_NAME, "variance_adaptive.cu");

		// Output buffer of adaptive post processing 
		//createAndPassBufferFromTo("per_window_variance_buffer_input", "per_window_variance_buffer_output", RT_FORMAT_FLOAT4, width, height, use_pbo);

		_context["sqrt_num_samples"]->setUint(_sqrt_num_samples);
		_context["num_samples"]->setUint(_num_samples);
		_context["bad_color"]->setFloat(1000000.0f, 0.0f, 1000000.0f); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
		_context["bg_color"]->setFloat(make_float3(0.0f));//0.390625f, 0.58203125f, 0.92578125f));

		_context["camera_changed"]->setInt(1);
	}

	void display(TrackballCamera* camera)
	{ 
		camera->update(_frameNumber);
		_context->launch(0, _width, _height);
		camera->setChanged(false);

		sutil::displayBufferGL(_getOutputBuffer());

		{
			static unsigned frame_count = 0;
			sutil::displayFps(frame_count++);
		}

		glutSwapBuffers();
	}

	void clean()
	{
		_destroyContext();
	}

	void saveToFile(std::string out_file)
	{
		_context->launch(0, _width, _height);
			
		sutil::displayBufferPPM(out_file.c_str(), _getOutputBuffer(), false);
	}

	Context getContext() { return _context; }

	Context* getContextPtr() { return &_context; }

protected:

	Context _context;

	uint32_t _width;
	uint32_t _height;

	int _frameNumber;
	int _sqrt_num_samples;
	int _num_samples;

private:

	// 'fromBufferName' buffer will be declared and created with the specified parameters and passed to 'toBufferName' buffer while it is declared at the same time.
	// The 'toBufferName' buffer will have the same specifications as the 'fromBufferName' buffer.
	void _createAndPassBufferFromTo(std::string fromBufferName, std::string toBufferName, RTformat bufferFormat, uint32_t width, uint32_t height, bool use_pbo)
	{
		Buffer buffer = sutil::createOutputBuffer(_context, bufferFormat, width, height, use_pbo);
		_context[fromBufferName]->set(buffer);

		_context->declareVariable(toBufferName)->set(_context[fromBufferName]->getBuffer());
	}

	void _passBufferFromTo(std::string fromBufferName, std::string toBufferName)
	{
		_context->declareVariable(toBufferName)->set(_context[fromBufferName]->getBuffer());
	}

	Buffer _getOutputBuffer()
	{
		return _context["output_buffer"]->getBuffer();
	}

	void _destroyContext()
	{
		if (_context)
		{
			_context->destroy();
			_context = 0;
		}
	}
};

//int currentInitialRenderNumSamples = 1;
//bool initalRenderDone = false;
//
//void setCurrentInitialRenderNumSamples(int initialRenderNumSamples)
//{
//	currentInitialRenderNumSamples = initialRenderNumSamples;
//}
//
//void updateCurrentInitialRenderNumSamples(Context context, bool cameraChanged, int maxNumSamplesPerRenderRun, int initialRenderNumSamples)
//{
//	if (currentInitialRenderNumSamples > 0)
//	{
//		int currentMaxNumSamplesPerRenderRun = min( currentInitialRenderNumSamples, maxNumSamplesPerRenderRun);
//		currentInitialRenderNumSamples -= currentMaxNumSamplesPerRenderRun;
//		initalRenderDone = false;
//		context["num_samples"]->setUint(currentMaxNumSamplesPerRenderRun);
//	}
//	else 
//	{
//		initalRenderDone = true;
//	}
//	if (cameraChanged)
//	{
//		initalRenderDone = false;
//		currentInitialRenderNumSamples = initialRenderNumSamples;
//	}
//}
//
//bool isInitialRenderRunDone() 
//{
//	return initalRenderDone;
//}