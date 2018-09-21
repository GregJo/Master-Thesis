/*
* Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
* optixHello.cpp -- Renders a solid green image.
*
* A filename can be given on the command line to write the results to file.
*/

/////////////////////////////////////////////////////////////////////////////

/*
Scientific questions for master thesis:
1. How to implement a general additional, adaptive ray launching mechanism in OptiX?
- host/API side:
- It has to be able to start an arbitary(!) additional ammount of rays depending on output(!), and/or if a condition(!) is met.
- I likely will not extend the given OptiX xpp-wrapper itself to implement my adaptive ray launching mechanism.
I intend to write my adaptive implementation into seperate source files, using the OptiX xpp-wrapper as a base.						(next ToDo)
- I probably have to add .png loading capability to the .obj loader used by the OptiX xpp-wrapper (nearly done)							(nearly done)
- Later on switch to and/or extend on the "optixPathTracer" project																		(next ToDo)
- device/.cu side:
- Implement it as an additional adaptive pass (must ensure that the first resulting image is completely avaible)						(basically done)
- Have a user set maximum sample budget per pixel (done in so far as i have a basic working implementation)								(done)
- Have a function that is providing current additional adaptive sample count, based on the neighborhood of the current launch index.	(next ToDo)
- In case of race conditions use atomics
- Launch addtional, adaptive rays with "rtTrace".																						(done)

2. Extra (device/.cu side): Make the adaptive pass dynamic, which means that the function that is providing current additional adaptive sample count will start,
as soon it has the necessary neighborhood values avaible. Disregarding whether the whole initial image is finished.
3. Extra (device/.cu side): Make use of the "visit" function to implement the adaptive ray pass on ray level, instead of the usual image space methods,
that require a first, initial result image.
*/

/////////////////////////////////////////////////////////////////////////////

/*
Understanding how to work with OptiX:
1. Find out how to load a 3D object.
- Done. I learned that I should be using and the given OptiX xpp-wrapper.
It is very compact and versatile in implementation already, meaning
it can handle the addition/change of custom components (like Programs) very well.
The wrapper even has fall backs to default programs, which are part of "sutil_sdk".

2. Find out how to access the (neighboring) output buffer values in the ray generation program for reading.
Will likely treat the initial input as a texture assigning a sampler to it.
3. Find out how to implement multiple passes.
- Done. Very similar to the post processing framework.

4. Find out how the progressive ray tracing example works.
5. Find out how the post processing framework works.
- Done. Thats what i use to implement the adaptive pass, namely the command list of the post processing framework.
In fact i don't really need it, but i will use it anyway.
I use the command list to implement an additional pass. All that the command list does is to ensure that each launch happens
in order as it was inserted after the previous has been completed.
The command list can simply be left out, as i could simply add launches of ray generation programs after the initial one without having to worry,
that the output of a previous is not complete, as it seems guaranteed that the program proceeds only after the output is ready.

6. Evaluate, whether multiple passes, progressive or post processing framework approach is suited, for adaptive ray launching. (an additional pass seems so far most appropriate)
extra: Find out whether a "dynanmic" adaptive ray launching is possible, i.e. necessary neighborhood of
the current 2D launch index in the ray generation program recieved the output values necessary for adaptive
ray launching and so the additional rays can be launched. For that i will need at least one more additional output buffer.
-
*/

/*
Concrete tasks before master thesis:
(Online - primary)
1. Research on related work on adaptive online rendering with path tracing.
2. Combine my current progress with realtime, online path tracing provided by the optix example and make it work.
3. Implement a simple variance based adaptive rendering algorithm on the realtime, online path tracing with adaptive post processing
(algorithm described in "Physically Based Rendering. From Theory to Implementation third edition", page 402). Make it work.

Important aspects to consider:
- Correctness
- Coherency
- Exploitation of time coherency (not much change from frame t to frame t+1)
- No artifacts allowed

Also already start writing on related works chapter.

(Offline - secondary)
Bonus:
Learn about the mechanism, that allows optix to decide how many rays will be sent at any collision (Based on the "visit" function
(determines in the BVH which nodes might be selected for the next visit), which operates on "selector nodes").

After that i should be able to make a proposal of the topic my master thesis will be about.
*/

/*
#include <optix.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>

#include "commonStructs.h"
#include "RayTraceRenderContext.h"

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <OptiXMesh.h>

#include <iomanip>

using namespace optix;

const char* const SAMPLE_NAME = "optixHello";

//
// Globals
//

Context        context;
uint32_t       width = 1024u;
uint32_t       height = 768u;
bool           use_pbo = false;
Aabb    aabb;

float3         camera_up;
float3         camera_lookat;
float3         camera_eye;

optix::float3 U, V, W;

void printUsageAndExit(const char* argv0);

// Postprocessing
CommandList commandListAdditionalRays;

//
// Predeclares
//

struct UsageReportLogger;

Buffer getOutputBuffer();
Buffer getPostProcessOutputBuffer();
void destroyContext();
void createContext(int usage_report_level, UsageReportLogger* logger);
void loadMesh(const std::string& filename);
void setupCamera();
void setupLights();

//
// Helper Functions
//

Buffer getOutputBuffer()
{
	return context["output_buffer"]->getBuffer();
}

Buffer getPostProcessOutputBuffer()
{
	return context["post_process_output_buffer"]->getBuffer();
}

void destroyContext()
{
	if (context)
	{
		context->destroy();
		context = 0;
	}
}


struct UsageReportLogger
{
	void log(int lvl, const char* tag, const char* msg)
	{
		std::cout << "[" << lvl << "][" << std::left << std::setw(12) << tag << "] " << msg;
	}
};

// Static callback
void usageReportCallback(int lvl, const char* tag, const char* msg, void* cbdata)
{
	// Route messages to a C++ object (the "logger"), as a real app might do.
	// We could have printed them directly in this simple case.

	UsageReportLogger* logger = reinterpret_cast<UsageReportLogger*>(cbdata);
	logger->log(lvl, tag, msg);
}

void registerExitHandler()
{
	// register shutdown handler
#ifdef _WIN32
	//glutCloseFunc(destroyContext);  // this function is freeglut-only
#else
	atexit(destroyContext);
#endif
}


void createContext(int usage_report_level, UsageReportLogger* logger)
{
	// Set up context
	context = Context::create();
	context->setRayTypeCount(2);
	//context->setEntryPointCount(2);
	context->setEntryPointCount(1);
	if (usage_report_level > 0)
	{
		context->setUsageReportCallback(usageReportCallback, usage_report_level, logger);
	}

	context["radiance_ray_type"]->setUint(0u);
	context["shadow_ray_type"]->setUint(1u);
	context["scene_epsilon"]->setFloat(1.e-6f);

	Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo);
	context["output_buffer"]->set(buffer);

	// Ray generation program
	const char *ptx = sutil::getPtxString(SAMPLE_NAME, "draw_color.cu");
	Program ray_gen_program = context->createProgramFromPTXString(ptx, "pinhole_camera");
	context->setRayGenerationProgram(0, ray_gen_program);

	// Exception program
	Program exception_program = context->createProgramFromPTXString(ptx, "exception");
	context->setExceptionProgram(0, exception_program);
	context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

	// Miss program
	context->setMissProgram(0, context->createProgramFromPTXString(sutil::getPtxString(SAMPLE_NAME, "draw_color.cu"), "miss"));
	context["bg_color"]->setFloat(0.2f, 0.2f, 0.2f);

	context->declareVariable("input_buffer")->set(getOutputBuffer());

	//// Output buffer of adaptive post processing 
	//Buffer post_process_out_buffer = sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo);
	//context["post_process_output_buffer"]->set(post_process_out_buffer);

	//// Adaptive ray generation program
	////const char *ptx = sutil::getPtxString(SAMPLE_NAME, "draw_color.cu");
	//Program adaptive_ray_gen_program = context->createProgramFromPTXString(ptx, "adaptive_camera");
	//context->setRayGenerationProgram(1, adaptive_ray_gen_program);										// Not sure if i need to set this as current ray generation program.
	//																									// Might need to reset this for the next loop initial render (ToDo).
	//																									// The reason i am doing it currently is that i try to reuse as much of the setup as possible as suggested.
	//																									// Exception program

	////Program exception_program = context->createProgramFromPTXString(ptx, "exception");
	//context->setExceptionProgram(1, exception_program);
	//context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);
}


void loadMesh(const std::string& filename)
{
	OptiXMesh mesh;
	mesh.context = context;
	//mesh.closest_hit = context->createProgramFromPTXString(sutil::getPtxString(SAMPLE_NAME, "draw_color.cu"), "closest_hit_radiance0");
	loadMesh(filename, mesh);

	aabb.set(mesh.bbox_min, mesh.bbox_max);

	GeometryGroup geometry_group = context->createGeometryGroup();
	geometry_group->addChild(mesh.geom_instance);
	geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
	context["top_object"]->set(geometry_group);
	context["top_shadower"]->set(geometry_group);
}

void setupCamera()
{
	const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components

	camera_eye = aabb.center() + make_float3(0.0f, 0.0f, max_dim*1.5f);
	camera_lookat = aabb.center();
	camera_up = make_float3(0.0f, 1.0f, 0.0f);

	const float vfov = 90.0f;
	const float aspect_ratio = static_cast<float>(width) /
		static_cast<float>(height);

	bool setCustomCameraValues = true;
	if (setCustomCameraValues)
	{
		camera_eye.x = 500.0f;
		camera_eye.y = 1000.0f;
		camera_eye.z = 0.0f;

		camera_lookat.x = 0.01f;
		camera_lookat.y = 0.01f;
		camera_lookat.z = 0.01f;

		camera_up.x = 0.0f;
		camera_up.y = 1.0f;
		camera_up.z = 0.0f;
	}

	sutil::calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
		U, V, W, true);

	context["eye"]->setFloat(camera_eye);
	context["U"]->setFloat(U);
	context["V"]->setFloat(V);
	context["W"]->setFloat(W);
}

void setupLights()
{
	const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components

	BasicLight lights[] = {
		{ make_float3(-0.5f,  0.25f, -1.0f), make_float3(0.2f, 0.2f, 0.25f), 0, 0 },
		{ make_float3(-0.5f,  0.0f ,  1.0f), make_float3(0.1f, 0.1f, 0.10f), 0, 0 },
		{ make_float3(0.5f,  0.5f ,  0.5f), make_float3(0.7f, 0.7f, 0.65f), 1, 0 }
	};
	lights[0].pos *= max_dim * 10.0f;
	lights[1].pos *= max_dim * 10.0f;
	lights[2].pos *= max_dim * 10.0f;

	Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(BasicLight));
	light_buffer->setSize(sizeof(lights) / sizeof(lights[0]));
	memcpy(light_buffer->map(), lights, sizeof(lights));
	light_buffer->unmap();

	context["lights"]->set(light_buffer);
}


void setupAdditionalRaysBuffer()
{
	unsigned char maxPerLaunchIdxRayBudget = static_cast<unsigned char>(5u);
	unsigned char* perLaunchIdxRayBudgets = new unsigned char[width * height * 4];

	// initialize additional rays buffer
	for (unsigned int i = 0; i < width * height; i++)
	{
		perLaunchIdxRayBudgets[i * 4] = static_cast<unsigned char>(static_cast<unsigned int>(maxPerLaunchIdxRayBudget) + 1u);
		perLaunchIdxRayBudgets[i * 4 + 1] = static_cast<unsigned char>(static_cast<unsigned int>(maxPerLaunchIdxRayBudget) + 1u);
		perLaunchIdxRayBudgets[i * 4 + 2] = static_cast<unsigned char>(static_cast<unsigned int>(maxPerLaunchIdxRayBudget) + 1u);
		perLaunchIdxRayBudgets[i * 4 + 3] = static_cast<unsigned char>(static_cast<unsigned int>(maxPerLaunchIdxRayBudget) + 1u);
	}

	// Additional rays test buffer setup
	Buffer additional_rays_buffer = sutil::createInputOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo);	// normally RT_FORMAT_UNSIGNED_BYTE would be enough, or RT_FORMAT_UNSIGNED_INT, 
																																// depending on the magnitude of additional samples one plans to send, 
																																// but i might want to visualize it
	memcpy(additional_rays_buffer->map(), perLaunchIdxRayBudgets, sizeof(unsigned char) * width * height * 4);
	additional_rays_buffer->unmap();
	context["additional_rays_buffer"]->set(additional_rays_buffer);

	delete[] perLaunchIdxRayBudgets;
}

/*
I want to find out if i can use the post processing framework without any inbuilt stages, but additional launches only.
The order as i imagine the post processing for additional, addaptive rays to happen:
1. Input image. (1. Ray gen program)
2. Compute additional ray count.
3. Launch additional rays and and merge the result into the input image of 1. step. (Switch to 2. ray gen program)
*/
/*
void setupPostprocessing()
{
	commandListAdditionalRays = context->createCommandList();

	// Input buffer for post processing
	setupAdditionalRaysBuffer();
	//context->declareVariable("input_buffer")->set(getOutputBuffer());

	//// Output buffer of adaptive post processing 
	//Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo);
	//context["post_process_output_buffer"]->set(buffer);

	//// Adaptive ray generation program
	//const char *ptx = sutil::getPtxString(SAMPLE_NAME, "draw_color.cu");
	//Program adaptive_ray_gen_program = context->createProgramFromPTXString(ptx, "adaptive_camera");
	//context->setRayGenerationProgram(1, adaptive_ray_gen_program);										// Not sure if i need to set this as current ray generation program.
	//																									// Might need to reset this for the next loop initial render (ToDo).
	//																									// The reason i am doing it currently is that i try to reuse as much of the setup as possible as suggested.
	//																									// Exception program
	//
	//Program exception_program = context->createProgramFromPTXString(ptx, "exception");
	//context->setExceptionProgram(1, exception_program);
	//context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

	commandListAdditionalRays->appendLaunch(1, width, height);
	commandListAdditionalRays->finalize();
}

int main(int argc, char* argv[])
{
	try {
		char outfile[1024];

		int width2 = width;
		int height2 = height;
		int i;

		outfile[0] = '\0';

		sutil::initGlut(&argc, argv);

		for (i = 1; i < argc; ++i) {
			if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
				printUsageAndExit(argv[0]);
			}
			else if (strcmp(argv[i], "--file") == 0 || strcmp(argv[i], "-f") == 0) {
				if (i < argc - 1) {
					strcpy(outfile, argv[++i]);
				}
				else {
					printUsageAndExit(argv[0]);
				}
			}
			else if (strncmp(argv[i], "--dim=", 6) == 0) {
				const char *dims_arg = &argv[i][6];
				sutil::parseDimensions(dims_arg, width2, height2);
			}
			else {
				fprintf(stderr, "Unknown option '%s'\n", argv[i]);
				printUsageAndExit(argv[0]);
			}
		}

		UsageReportLogger logger;

		//
		// Ray trace renderer test
		//
		BasicLight lights[] = {
			{ make_float3(-0.5f,  0.25f, -1.0f), make_float3(0.2f, 0.2f, 0.25f), 0, 0 },
			{ make_float3(-0.5f,  0.0f ,  1.0f), make_float3(0.1f, 0.1f, 0.10f), 0, 0 },
			{ make_float3(0.5f,  0.5f ,  0.5f), make_float3(0.7f, 0.7f, 0.65f), 1, 0 }
		};

		RayTraceRenderContext rayTracertContext(SAMPLE_NAME, width, height);
		//std::shared_ptr<IOptixRenderContext> rayTracertContext = new RayTraceRenderContext(SAMPLE_NAME, width, height);

		rayTracertContext.setupContext(0, &logger, false);
		unsigned int numberOfLights = sizeof(lights) / sizeof(lights[0]);
		rayTracertContext.setupScene(camera_eye, camera_up, camera_lookat, lights, numberOfLights, "../bin/Data/sponza/sponza.obj");
		rayTracertContext.display(outfile, argv[0]);

		//
		// Ray trace renderer test end
		//

		//createContext(0, &logger);

		//loadMesh("../bin/Data/sponza/sponza.obj");
		//setupCamera();

		//setupLights();

		////setupAdditionalRaysBuffer();

		////setupPostprocessing();

		//      /* Run */
		//context->validate();
		//context->launch(0, width, height);

		////commandListAdditionalRays->execute();

		//      /* Display image */
		//      if( strlen( outfile ) == 0 ) {
		//          sutil::displayBufferGlut( argv[0], getOutputBuffer() );
		//      } else {
		//          sutil::displayBufferPPM( outfile, getOutputBuffer(), false);
		//      }

		////if (strlen(outfile) == 0) {
		////	sutil::displayBufferGlut(argv[0], getPostProcessOutputBuffer());
		////}
		////else {
		////	sutil::displayBufferPPM(outfile, getPostProcessOutputBuffer(), false);
		////}

		//destroyContext();

		//delete[] perLaunchIdxRayBudgets;
/*
		return(0);

	} SUTIL_CATCH(context->get())
}


void printUsageAndExit(const char* argv0)
{
	fprintf(stderr, "Usage  : %s [options]\n", argv0);
	fprintf(stderr, "Options: --file | -f <filename>      Specify file for image output\n");
	fprintf(stderr, "         --help | -h                 Print this usage message\n");
	fprintf(stderr, "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n");
	exit(1);
}
*/