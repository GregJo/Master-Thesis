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
Currently i stand before one very obvious problem in my programming task, one simply must not read and write from/into the one and same buffer 
when working with multiple threads, which is obvious considering that i am working on the GPU.
Right now i can think of two solutions:
	1. I may "simply" do an additional pass computing the addtional adaptive sample map for the next render pass.
	   Here i have to be very careful to only compute and write for only one launch index/pixel of a small window, for the whole small window,
	   in order to avoid race conditions and multiple write accesses, which would cause the program to crash (which should be rather easy, 
	   if guarantee to do it only on the center pixels of the small windows).
	   With this mehtod i would have the advantage of doing it once for the whole adaptive samples map.
	   The disadvantage would be having another pass, which might be "unnecessarily" imperformant.
	   Also the method might not work if my neighborhood is not exactly known beforehand or not following an easy scheme (like the window neighborhood 
	   i already mentioned, in a simple case).
	2. I do redundant computations before launching each and every ray based on the input, which i would access read only. 
	   The advantage is not having to take care of parallel simultanious write access handling.
	   Another advantage is that i can handle more complex methods with an unknown neighborhood size and shape.
	   The disadvantage is that i have to do redundant computations for lets say a window neighborhood, which might be even more imperformant
	   than the additional pass, but not sure, due to highly parallel nature of GPU.
*/

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

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include "optixPathTracer.h"
#include <sutil.h>
#include <Arcball.h>

#include <OptiXMesh.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <iomanip>

#include "TrackballCamera.h"

using namespace optix;

const char* const SAMPLE_NAME = "optixHello";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context        context = 0;
uint32_t       width = 512;
uint32_t       height = 512;
bool           use_pbo = true;

TrackballCamera* camera;

//bool initial_render_run = true;

int            frame_number = 1;
int            sqrt_num_samples = 1;
int            rr_begin_depth = 1;
Program        pgram_intersection = 0;
Program        pgram_bounding_box = 0;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
bool           camera_changed = true;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

// Postprocessing
bool usePostProcessing = false;
CommandList commandListAdaptive;

//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
Buffer getPostProcessOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void loadGeometry();
//void setupCamera();
//void updateCamera();
void glutInitialize(int* argc, char** argv);
void glutRun();

void glutDisplay();
void glutKeyboardPress(unsigned char k, int x, int y);
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);
void glutResize(int w, int h);


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

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


void registerExitHandler()
{
	// register shutdown handler
#ifdef _WIN32
	glutCloseFunc(destroyContext);  // this function is freeglut-only
#else
	atexit(destroyContext);
#endif
}


void setMaterial(
	GeometryInstance& gi,
	Material material,
	const std::string& color_name,
	const float3& color)
{
	gi->addMaterial(material);
	gi[color_name]->setFloat(color);
}


GeometryInstance createParallelogram(
	const float3& anchor,
	const float3& offset1,
	const float3& offset2)
{
	Geometry parallelogram = context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	parallelogram->setIntersectionProgram(pgram_intersection);
	parallelogram->setBoundingBoxProgram(pgram_bounding_box);

	float3 normal = normalize(cross(offset1, offset2));
	float d = dot(normal, anchor);
	float4 plane = make_float4(normal, d);

	float3 v1 = offset1 / dot(offset1, offset1);
	float3 v2 = offset2 / dot(offset2, offset2);

	parallelogram["plane"]->setFloat(plane);
	parallelogram["anchor"]->setFloat(anchor);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);

	GeometryInstance gi = context->createGeometryInstance();
	gi->setGeometry(parallelogram);
	return gi;
}


void createContext()
{
	context = Context::create();
	context->setRayTypeCount(2);
	//context->setEntryPointCount(1);
	context->setEntryPointCount(2);
	context->setStackSize(1800);

	context->setPrintEnabled(true);
	context->setPrintBufferSize(2048);

	context["scene_epsilon"]->setFloat(1.e-3f);
	context["pathtrace_ray_type"]->setUint(0u);
	context["pathtrace_shadow_ray_type"]->setUint(1u);
	context["rr_begin_depth"]->setUint(rr_begin_depth);

	Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
	context["output_buffer"]->set(buffer);

	// Setup programs
	const char *ptx = sutil::getPtxString(SAMPLE_NAME, "optixPathTracer.cu");
	context->setRayGenerationProgram(0, context->createProgramFromPTXString(ptx, "pathtrace_camera"));
	context->setExceptionProgram(0, context->createProgramFromPTXString(ptx, "exception"));
	context->setMissProgram(0, context->createProgramFromPTXString(ptx, "miss"));

	context->declareVariable("input_buffer")->set(getOutputBuffer());

	// Output buffer of adaptive post processing 
	Buffer post_process_out_buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
	context["post_process_output_buffer"]->set(post_process_out_buffer);

	Program adaptive_ray_gen_program = context->createProgramFromPTXString(ptx, "pathtrace_camera_adaptive");
	context->setRayGenerationProgram(1, adaptive_ray_gen_program);		

	context["sqrt_num_samples"]->setUint(sqrt_num_samples);
	context["bad_color"]->setFloat(1000000.0f, 0.0f, 1000000.0f); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
	context["bg_color"]->setFloat(make_float3(0.0f));
}


void loadGeometry()
{
	// Light buffer
	ParallelogramLight light;
	light.corner = make_float3(343.0f, 548.6f, 227.0f);
	light.v1 = make_float3(-130.0f, 0.0f, 0.0f);
	light.v2 = make_float3(0.0f, 0.0f, 105.0f);
	light.normal = normalize(cross(light.v1, light.v2));
	light.emission = make_float3(15.0f, 15.0f, 5.0f);

	Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(ParallelogramLight));
	int b = sizeof(ParallelogramLight);
	light_buffer->setSize(1u);
	int a = sizeof(light);
	memcpy(light_buffer->map(), &light, sizeof(light));
	light_buffer->unmap();
	context["lights"]->setBuffer(light_buffer);


	// Set up material
	Material diffuse = context->createMaterial();
	const char *ptx = sutil::getPtxString(SAMPLE_NAME, "optixPathTracer.cu");
	Program diffuse_ch = context->createProgramFromPTXString(ptx, "diffuse");
	//Program diffuse_ch = context->createProgramFromPTXString(ptx, "diffuse_textured");
	Program diffuse_ah = context->createProgramFromPTXString(ptx, "shadow");
	diffuse->setClosestHitProgram(0, diffuse_ch);
	diffuse->setAnyHitProgram(1, diffuse_ah);

	Material diffuse_light = context->createMaterial();
	Program diffuse_em = context->createProgramFromPTXString(ptx, "diffuseEmitter");
	diffuse_light->setClosestHitProgram(0, diffuse_em);

	// Set up parallelogram programs
	ptx = sutil::getPtxString(SAMPLE_NAME, "parallelogram.cu");
	pgram_bounding_box = context->createProgramFromPTXString(ptx, "bounds");
	pgram_intersection = context->createProgramFromPTXString(ptx, "intersect");

	// create geometry instances
	std::vector<GeometryInstance> gis;

	const float3 white = make_float3(0.8f, 0.8f, 0.8f);
	const float3 green = make_float3(0.05f, 0.8f, 0.05f);
	const float3 red = make_float3(0.8f, 0.05f, 0.05f);
	const float3 light_em = make_float3(15.0f, 15.0f, 5.0f);

	// Floor
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, 559.2f),
		make_float3(556.0f, 0.0f, 0.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);

	// Ceiling
	gis.push_back(createParallelogram(make_float3(0.0f, 548.8f, 0.0f),
		make_float3(556.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, 559.2f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);

	// Back wall
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 559.2f),
		make_float3(0.0f, 548.8f, 0.0f),
		make_float3(556.0f, 0.0f, 0.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);

	// Right wall
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
		make_float3(0.0f, 548.8f, 0.0f),
		make_float3(0.0f, 0.0f, 559.2f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", green);

	// Left wall
	gis.push_back(createParallelogram(make_float3(556.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, 559.2f),
		make_float3(0.0f, 548.8f, 0.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", red);

	// Short block
	gis.push_back(createParallelogram(make_float3(130.0f, 165.0f, 65.0f),
		make_float3(-48.0f, 0.0f, 160.0f),
		make_float3(160.0f, 0.0f, 49.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(290.0f, 0.0f, 114.0f),
		make_float3(0.0f, 165.0f, 0.0f),
		make_float3(-50.0f, 0.0f, 158.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(130.0f, 0.0f, 65.0f),
		make_float3(0.0f, 165.0f, 0.0f),
		make_float3(160.0f, 0.0f, 49.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(82.0f, 0.0f, 225.0f),
		make_float3(0.0f, 165.0f, 0.0f),
		make_float3(48.0f, 0.0f, -160.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(240.0f, 0.0f, 272.0f),
		make_float3(0.0f, 165.0f, 0.0f),
		make_float3(-158.0f, 0.0f, -47.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);

	// Tall block
	gis.push_back(createParallelogram(make_float3(423.0f, 330.0f, 247.0f),
		make_float3(-158.0f, 0.0f, 49.0f),
		make_float3(49.0f, 0.0f, 159.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(423.0f, 0.0f, 247.0f),
		make_float3(0.0f, 330.0f, 0.0f),
		make_float3(49.0f, 0.0f, 159.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(472.0f, 0.0f, 406.0f),
		make_float3(0.0f, 330.0f, 0.0f),
		make_float3(-158.0f, 0.0f, 50.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(314.0f, 0.0f, 456.0f),
		make_float3(0.0f, 330.0f, 0.0f),
		make_float3(-49.0f, 0.0f, -160.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	gis.push_back(createParallelogram(make_float3(265.0f, 0.0f, 296.0f),
		make_float3(0.0f, 330.0f, 0.0f),
		make_float3(158.0f, 0.0f, -49.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);

	// Create shadow group (no light)
	GeometryGroup shadow_group = context->createGeometryGroup(gis.begin(), gis.end());
	shadow_group->setAcceleration(context->createAcceleration("Trbvh"));
	context["top_shadower"]->set(shadow_group);

	// Light
	gis.push_back(createParallelogram(make_float3(343.0f, 548.6f, 227.0f),
		make_float3(-130.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, 105.0f)));
	setMaterial(gis.back(), diffuse_light, "emission_color", light_em);

	// Create geometry group
	GeometryGroup geometry_group = context->createGeometryGroup(gis.begin(), gis.end());
	geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
	context["top_object"]->set(geometry_group);
}

void loadComplexGeometry()
{
	// set up material
	Material diffuse = context->createMaterial();
	const char *ptx = sutil::getPtxString(SAMPLE_NAME, "optixPathTracer.cu");
	Program diffuse_ch = context->createProgramFromPTXString(ptx, "diffuseTextured");
	Program diffuse_ah = context->createProgramFromPTXString(ptx, "shadow");
	diffuse->setClosestHitProgram(0, diffuse_ch);
	diffuse->setAnyHitProgram(1, diffuse_ah);

	Material diffuse_light = context->createMaterial();
	Program diffuse_em = context->createProgramFromPTXString(ptx, "diffuseEmitter");
	diffuse_light->setClosestHitProgram(0, diffuse_em);

	// create geometry instances

	// load model
	Aabb model_aabb;
	OptiXMesh mesh;

	mesh.closest_hit = diffuse_ch;
	mesh.any_hit = diffuse_ah;

	mesh.context = context;
	const std::string filename = "../bin/Data/sponza/sponza.obj";
	loadMesh(filename, mesh);

	model_aabb.set(mesh.bbox_min, mesh.bbox_max);

	GeometryGroup geometry_group = context->createGeometryGroup();
	geometry_group->addChild(mesh.geom_instance);
	geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
	context["top_object"]->set(geometry_group);
	context["top_shadower"]->set(geometry_group);

	const float3 white = make_float3(0.8f, 0.8f, 0.8f);

	// Setup diffuse textures (Kd_Map)

	// Light
	const float3 light_em = make_float3(15.0f, 15.0f, 5.0f);

	// Light buffer
	ParallelogramLight light;
	light.corner = make_float3(model_aabb.center().x + 0.0f, model_aabb.center().y + 100.0f, model_aabb.center().z + 50.0f);
	light.v1 = make_float3(200.0f, 0.0f, 0.0f);
	light.v2 = make_float3(0.0f, 0.0f, -200.0f);
	light.normal = normalize(cross(light.v1, light.v2));
	light.emission = make_float3(15.0f, 15.0f, 5.0f);

	Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(ParallelogramLight));
	int b = sizeof(ParallelogramLight);
	light_buffer->setSize(1u);
	int a = sizeof(light);
	memcpy(light_buffer->map(), &light, sizeof(light));
	light_buffer->unmap();
	context["lights"]->setBuffer(light_buffer);

	GeometryInstance light_parallelogram = createParallelogram(make_float3(model_aabb.center().x + 0.0f, model_aabb.center().y + 100.0f, model_aabb.center().z + 50.0f),
		make_float3(200.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, -200.0f));
	setMaterial(light_parallelogram, diffuse_light, "emission_color", light_em);

	// Create geometry group
}

//
// Post Processing begin
//

void setupAdditionalRaysBuffer()
{
	unsigned char maxPerLaunchIdxRayBudget = static_cast<unsigned char>(2u);
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
	Buffer additional_rays_buffer = sutil::createInputOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, width, height, false);	
	memcpy(additional_rays_buffer->map(), perLaunchIdxRayBudgets, sizeof(unsigned char) * width * height * 4);
	additional_rays_buffer->unmap();
	context["additional_rays_buffer"]->set(additional_rays_buffer);

	delete[] perLaunchIdxRayBudgets;
}

void setupPostprocessing()
{
	commandListAdaptive = context->createCommandList();

	// Input buffer for post processing
	//setupAdditionalRaysBuffer();

	commandListAdaptive->appendLaunch(1, width, height);
	commandListAdaptive->finalize();
	usePostProcessing = true;
}

//
// Post Processing end
//

//void setupCamera()
//{
//	camera_eye = make_float3(-500.0f, 1250.0f, 0.0f);
//	camera_lookat = make_float3(0.0f, 0.0f, 0.0f);
//	camera_up = make_float3(0.0f, 1.0f, 0.0f);
//
//	camera_rotate = Matrix4x4::identity();
//}
//
//
//void updateCamera()
//{
//	const float fov = 35.0f;
//	const float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
//
//	float3 camera_u, camera_v, camera_w;
//	sutil::calculateCameraVariables(
//		camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
//		camera_u, camera_v, camera_w, /*fov_is_vertical*/ true);
//
//	const Matrix4x4 frame = Matrix4x4::fromBasis(
//		normalize(camera_u),
//		normalize(camera_v),
//		normalize(-camera_w),
//		camera_lookat);
//	const Matrix4x4 frame_inv = frame.inverse();
//	// Apply camera rotation twice to match old SDK behavior
//	const Matrix4x4 trans = frame*camera_rotate*camera_rotate*frame_inv;
//
//	camera_eye = make_float3(trans*make_float4(camera_eye, 1.0f));
//	camera_lookat = make_float3(trans*make_float4(camera_lookat, 1.0f));
//	camera_up = make_float3(trans*make_float4(camera_up, 0.0f));
//
//	sutil::calculateCameraVariables(
//		camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
//		camera_u, camera_v, camera_w, true);
//
//	camera_rotate = Matrix4x4::identity();
//
//	if (camera_changed) // reset accumulation
//		frame_number = 1;
//	camera_changed = false;
//
//	context["frame_number"]->setUint(frame_number++);
//	context["eye"]->setFloat(camera_eye);
//	context["U"]->setFloat(camera_u);
//	context["V"]->setFloat(camera_v);
//	context["W"]->setFloat(camera_w);
//
//}


void glutInitialize(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(SAMPLE_NAME);
	glutHideWindow();
}


void glutRun()
{
	// Initialize GL state                                                            
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, width, height);

	glutShowWindow();
	glutReshapeWindow(width, height);

	// register glut callbacks
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutDisplay);
	glutReshapeFunc(glutResize);
	glutKeyboardFunc(glutKeyboardPress);
	glutMouseFunc(glutMousePress);
	glutMotionFunc(glutMouseMotion);

	registerExitHandler();

	glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
	//updateCamera();
	camera->update(frame_number);
	context->launch(0, width, height);

	if (usePostProcessing)
	{
		commandListAdaptive->execute();
		sutil::displayBufferGL(getPostProcessOutputBuffer());
	}
	else
	{
		sutil::displayBufferGL(getOutputBuffer());
	}

	{
		static unsigned frame_count = 0;
		sutil::displayFps(frame_count++);
	}

	//initial_render_run = false;

	glutSwapBuffers();
}

void glutKeyboardPress(unsigned char k, int x, int y)
{

	switch (k)
	{
	case('q'):
	case(27): // ESC
	{
		destroyContext();
		exit(0);
	}
	case('s'):
	{
		const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
		std::cerr << "Saving current frame to '" << outputImage << "'\n";
		sutil::displayBufferPPM(outputImage.c_str(), getOutputBuffer(), false);
		break;
	}
	}
}


void glutMousePress(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_button = button;
		mouse_prev_pos = make_int2(x, y);
	}
	else
	{
		// nothing
	}
}


//void glutMouseMotion(int x, int y)
//{
//	if (mouse_button == GLUT_RIGHT_BUTTON)
//	{
//		const float dx = static_cast<float>(x - mouse_prev_pos.x) /
//			static_cast<float>(width);
//		const float dy = static_cast<float>(y - mouse_prev_pos.y) /
//			static_cast<float>(height);
//		const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
//		const float scale = std::min<float>(dmax, 0.9f);
//		camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
//		camera_changed = true;
//	}
//	else if (mouse_button == GLUT_LEFT_BUTTON)
//	{
//		const float2 from = { static_cast<float>(mouse_prev_pos.x),
//			static_cast<float>(mouse_prev_pos.y) };
//		const float2 to = { static_cast<float>(x),
//			static_cast<float>(y) };
//
//		const float2 a = { from.x / width, from.y / height };
//		const float2 b = { to.x / width, to.y / height };
//
//		camera_rotate = arcball.rotate(b, a);
//		camera_changed = true;
//	}
//
//	mouse_prev_pos = make_int2(x, y);
//}

void glutMouseMotion(int x, int y)
{
	if (mouse_button == GLUT_RIGHT_BUTTON)
	{
		const float dx = static_cast<float>(x - mouse_prev_pos.x) /
			static_cast<float>(width);
		const float dy = static_cast<float>(y - mouse_prev_pos.y) /
			static_cast<float>(height);
		const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
		const float scale = std::min<float>(dmax, 0.9f);
		//camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
		camera->setEye(camera->getEye() + (camera->getLookat() - camera->getEye()) * scale);
		//camera_changed = true;
		camera->setChanged(true);
	}
	else if (mouse_button == GLUT_LEFT_BUTTON)
	{
		const float2 from = { static_cast<float>(mouse_prev_pos.x),
			static_cast<float>(mouse_prev_pos.y) };
		const float2 to = { static_cast<float>(x),
			static_cast<float>(y) };

		const float2 a = { from.x / width, from.y / height };
		const float2 b = { to.x / width, to.y / height };

		//camera_rotate = arcball.rotate(b, a);
		camera->setRotation(arcball.rotate(b, a));
		//camera_changed = true;
		camera->setChanged(true);
	}

	mouse_prev_pos = make_int2(x, y);
}

void glutResize(int w, int h)
{
	if (w == (int)width && h == (int)height) return;

	//camera_changed = true;
	camera->setChanged(true);

	width = w;
	height = h;

	sutil::resizeBuffer(getOutputBuffer(), width, height);

	glViewport(0, 0, width, height);

	glutPostRedisplay();
}

void printUsageAndExit( const char* argv0 );

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

int main(int argc, char* argv[])
{
	std::string out_file;
	std::string mesh_file = std::string(sutil::samplesDir()) + "/data/cow.obj";
	for (int i = 1; i<argc; ++i)
	{
		const std::string arg(argv[i]);

		if (arg == "-h" || arg == "--help")
		{
			printUsageAndExit(argv[0]);
		}
		else if (arg == "-f" || arg == "--file")
		{
			if (i == argc - 1)
			{
				std::cerr << "Option '" << arg << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
			}
			out_file = argv[++i];
		}
		else if (arg == "-n" || arg == "--nopbo")
		{
			use_pbo = false;
		}
		else if (arg == "-m" || arg == "--mesh")
		{
			if (i == argc - 1)
			{
				std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
				printUsageAndExit(argv[0]);
			}
			mesh_file = argv[++i];
		}
		else
		{
			std::cerr << "Unknown option '" << arg << "'\n";
			printUsageAndExit(argv[0]);
		}
	}

    try { 
		glutInitialize(&argc, argv);

#ifndef __APPLE__
		glewInit();
#endif

		createContext();

		camera = new TrackballCamera(context, (int)width, (int)height);
		camera->setup(make_float3(-500.0f, 1250.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), 35.0f, true);

		//setupCamera();
		loadComplexGeometry();

		// Adaptive post processing setup
		setupPostprocessing();

		context->validate();

		if (out_file.empty())
		{
			glutRun();
		}
		else
		{
			//updateCamera();
			context->launch(0, width, height);
			// Adaptive test execute if active
			if (usePostProcessing)
			{
				commandListAdaptive->execute();
				sutil::displayBufferPPM(out_file.c_str(), getPostProcessOutputBuffer(), false);
			}
			else
			{
				sutil::displayBufferPPM(out_file.c_str(), getOutputBuffer(), false);
			}
			destroyContext();
		}

        return( 0 );

    } SUTIL_CATCH( context->get() )
}


void printUsageAndExit( const char* argv0 )
{
  fprintf( stderr, "Usage  : %s [options]\n", argv0 );
  fprintf( stderr, "Options: --file | -f <filename>      Specify file for image output\n" );
  fprintf( stderr, "         --help | -h                 Print this usage message\n" );
  fprintf( stderr, "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n" );
  exit(1);
}