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

/*
At the moment i implemented the very basic hoelder regularity computation of the alpha on a tile, of which i implemented the smooth regime as well as the non-smooth regime.
The computations are given by the paper and for the gradient, which is required by the smooth regime i used finite differences.
Right now all seems to work as intended except the values do not make any sense because of the logarithms involved the computed alpha often turns out to be negative, which should
not be the case as far as i understand. Another problem involving the logarithms is that log|x_center - x_neighbor|(value), which has to be computed as follows, again as i understand it:
log(value) / log(x_center - x_neighbor), which in turn might result in the denominator being zero if log(1).
The biggest uncertainty regarding applying the hoelder regularity computations of the alpha is that i am not sure which routine gets applied where and by which rules.
My current thought is to apply the non-smooth routine and if the alpha results being > 1.0 then apply the smooth routine to recompute it.

I considered the thought that i will have geometry information available, meaning knowing when one object ends and another begins in the current view/screen, which would be the areas
i would apply the non-smooth regime at and the smooth regime otherwise. 
Maybe the point of knowing the scene but still wanting to compute the regularity is to compute the alpha, which then tells you how many samples are needed.

Q1: Also the calculation of the samples needed looks to me rediculously huge as described by the paper: 
64 * 64 (64 * 64 wavelet basis functions at level 6 of the wavelet (binary) tree) * hoelder_alpha (probaly between 0.0 and 2.0) * 1.25 (oversampling factor).

I currenly lack understanding of:
	- how the wavelets are exactly used for reconstruction
*/

/*
A1: My current task involves implementing the hoelder alpha routine on multiple levels. Per descended level i add an additional addaptivive sample to be processed 
times the oversampling factor 1.25.

Before that i must ensure that i use the geometry information avaible to compute the gradients necessary for the smooth regime hoelder alpha computation.

Q1: Was i suggested to remove the fabric from the scene, like the hanging 'carpets' because they were very irregularly textured thus introducing another source for irregularity
in the scene making the hoelder regularity undreliable?

A1: Goal is simply a less complex scene.

Q2: Would it be better to compute the gradients based on the samples before the image reconstruction, as in more correct? Otherwise it sounded like it was a more complex, 
computationally expensive approach (correct me if i'm wrong here).

It's important to compute [X] on every level with the updated information. (forgot, what [X] exactly was, but it's important).
I believe it was regarding the finite differences, which will become more and more refined with each adaptive sample. 
-> Update the depth map and use the most current depth information avaible to compute gradient via finite differences.
*/

/*
Most important thing i learned regarding using the current gradient values: it is actually only the current level gradient information derived from the samples only obtained 
adaptively at the current level, disregarding all previously evaluated samples for the previous level gradients. This might not work for my current implementation though.

Another important point to consider is how super pixels are represented regarding the sampling. 
Currently if i say that i use one sample per pixel in a 64x64 window/super pixel i mean that i have 64x64 samples, one for each pixel of that window/super pixel. 
In Lessig's implementation it means having only one sample per pixel the same as having one sample per e.g. 64x64 window/super pixel. 

Regarding Lessigs implementation i also need to try out atomics in order to strictly avoid mulltiple computations of the window/super pixel representing color value/values 
as opposed to what i have now, allowing to compute the same value and writing it until following threads "notice" that it is already computed.

Regarding Lessigs C++ code: the most intresting/relevant part to me should be found in the utilities folder and the hoelder integrator/sampler (not sure anymore here).

My current task still stands from last meeting: implement multi-resolution hoelder alpha computation. 
Also as it currently stands this path tracing implementation will be progressive (i don't see a way yet to evaluate all the levels within one frames time).
Important things to consider:
	- I need an accumulation buffer for the additional adaptive samples. It should be updated as follow: if whithin one frame the ACCUMULATED number of additional, 
	  adaptive samples is greater than what i allow to evaluate whithin the time of one frame, then update the number according to 'currently total avaible samples 
	  minus the maximum allowed number of samples per ray used' and CARRY OVER the remaining samples to the next frames render run. 
	  Repeat until all accumulated, addtional, adaptive samples depleted.
	- I need a specific buffer/map to keep track of all pixel that belong to an area, which should be refined/need a finer level of resolution. Indicate all pixel in an are that 
	  needs to be refined with a negative value. Reset a refined area pxels value right after refinement at current, finer level.
	- I need a variable to keep track of how many levels i already descended. The finest level is reached if the area has a size of 2x2, or probably 3x3 pixels.
	- I need a variable that keeps track of the current area size at the current in (pixel)x(pixel). with eac level descended the window size is halved.
*/

/*
I now basically completed the current task and now i need to make it work correctly. As for now very homogenious areas seem to be refined to the finest level, 
meaning the hoelder alpha is close to zero. The opposite should be the case. So i have to make sure, that the hoelder alpha comuptation is correct.

Update: 
I am now more confident that the hoelder alpha computation is and was correct, the only thing bothering me now is that it is very small 
(to be "effective" the computed hoelder alpha has to be at least multiplied by a factor of hundred).

Also i found out that atomics most likely will not result in performance gain, as i tried this out on the adaptive variance based implementation, 
assuming i am using atomics correctly. Maybe setting the buffer type to 'RT_BUFFER_INPUT_OUTPUT' and using the flag 'RT_BUFFER_GPU_LOCAL' 
(parameter 'type' = RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL) might be worth trying, since it creates on each device a local copy of each buffer 
(in my case there is only one) and thus may actually be totally not for my use case, because of having only one GPU device anyway.
Another thought is, that atomics might be totally useless in my case, as the other 'GPU threads' still request an atomic exchange in my case and thus 
the exchanges will still be executed just as many times as without atomics.
*/

/*
Implementation:

Current most important implementation task: Mitchel filter for reconstruction.
I am uncertain on how to keep the samples on my 'film' separate as for now they are added and linearly with the inverse of the frame number, as weight between the previously 
computed and the currently computed MC sample, at each and every pixel. I need the seperate samples in order to apply a filter of a certain support/extent, to reconstruct a pixels value.

I need to take care of oversampling at areas where the luminosity is low scaling the number of addtional samples by said luminosity. Another related issue is currently that at places
where a MC sample ray completely misses the geometry the maximum amount of adaptive samples is applied, due to initial values certain helper buffers are filled with (easy to fix).

The most urgent task though is the further refactoring and restructuring of the code at host and device side. At device side i require cleaner sepration of initial and adaptive sampling
as well as a better separation of of different adaptive methods, which are currently the variance (!) based and the hoelder adaptive (!) implemetation (put them into seprate files).
The next step is to extract a first kind of an adaptive framework on device side. 
After that i will implement a MC renderer and extend it by a first adaptive MC rendering kind of framework.

This is not really an implementation task, but i have to find at least two more scenes for proper evaluation and comparison, also i need to consider well suited scenes for 
adaptive rendring (consisting of large regular areas/less complex geometry) as well as less suited scenese as i currently have one with the crytek sponza scene.

One more thing to consider regarding the hoelder implementation is that it is possible to achieve correctly weighted hoelder regularity and accordingly adaptive sample number results by
scaling them with the light source intensity (maybe also consider the distance from the light source when weighting).

Consider importance sampling when implementing specular/glossy MC rendering model.


Master Thesis:

Write a first version of "Introduction" chapter.
Read up on state of the art report about adaptive MC rendering and reconstruction and begin writing "Related Work" chapter.

Methodology:
"Equal time" and "equal quality" comparison between offline and online "Hoelder Adaptive Image Sythesis".  

Update:
My refactoring of device side adaptive MC implementations seems to be effective as it does not affect performance (currently only frod variance based adaptive MC implementation).
I can do better by passing input buffers needed for computations and move implementation specific buffers to corresponding implementations, instead of having them clutter my 
main adaptive postprocess file.

Update 2:
I attempted to base the refactoring, host- and device-side of my code on using the possibility to access buffers by ID on device side. 

It did not work bacause i apparently do not fully understand how to access multiple buffers in the buffer holding all buffersby ID. 
Another point is that only buffers of the same format may be grouped, which still might be worth it considering that i have many buffers of the same type.
In addtion i had problems implementing to get a buffer that i bound to the buffer of buffer IDs, making the buffers accessible by their ID, to pass it to another buffer 
(which i do for looping, implementing a cumulative depth buffer).
*/

/*
Implementation todo's/goals (including priority, 19.08.2018):

1. Implement a model/scene struct allowing for all configuration, as position of the geometry, lighting, scale, etc. 
(goal is to match pbrt scene (most importantly camera and light) settings for comparison).
2. Adjust .obj loader to correctly load materials and textures (currently dismisses all textures if not complete texture coverage for all the meshes of the model).
3. Implement basic brdf, so that at least glass and specularity can be handled.
4. Implement a sampler.
5. Move implementation closer to the paper it is supposed to be based on (correct gradient computation, building up wavelet tree, using wavelets for reconstruction (includes solving of linear equations)).
6. Get finally rid of glut and replace it with glfw. After that further abstraction will become possible/easier, eliminating all global variables.
7. Set up a cmake project, ideally with Unix compatibility.

An error to fix:
When reducing window size by half when computing hoelder make sure that the four new widow centre points are (0.25 * windowSize, 0.25 * windowSize), (0.75 * windowSize, 0.25 * windowSize),
(0.25 * windowSize, 0.75 * windowSize) and (0.75 * windowSize, 0.75 * windowSize). It will resolve the issue with the more or less tiny areas that stay undetailed, inbetween highly detailed areas.
Basically overhaul the centre point calculation for the hoelder computation in a certain window with the specific centre point (take care of correct inital setting, out of bounds and correct centre point refinement,
as described above). 

Also make sure i expend the current level samples before moving on with the hoelder computation for the next level.

If not all the samples that were added in the current adaptive level were expended make sure that the current adaptive level counter on host side goes up only when all current level, 
additional, adaptive samples were used.

Question:
What to do for reconstruction of the depthbuffer, average it by the total sample count (as it is currently the case), use mitchell filter or something else entirely?

Implementation Update (29.08.2018):
I noticed that the implementation of my output values at each adaptive level is not correct. I do not have any super pixels at all. I always work on a pixel level, whereas i need to work at the 
superpixel/adaptive window size level. 
Solve this with an additional adaptive buffer that holds the value for each super pixel/adaptive window at its current centre pixel. Clear this additional buffer each adaptive level.
Correct implementation should also lead to reduced computation costs.

Implement a new inital render pass method/ray generation program that operates on adaptive level super pixel/adaptive windows, where the centre pixel level represents all pixel of the current 
super pixel/adaptive level.

I probably need to reimplement all the hoelder adaptive methods in order to ensure correctness and consistency with the super pixel/adaptive windows. 

(02.09.2018) If an area of the image was excluded early by the hoelder regularity criteria, should i consider it at later levels again (currently i do not)? 

(06.09.2018) Implementation tasks:

1. Make sure Mitchell-Filter is working properly, known issue with unchanging sample origin values used for filter table value lookup.
2. Optix on Linux, for testing purposes.
3. Make cmake project of my implementation in general to be able to test it everywhere ideally.
4. Implement super pixel according to current level, so that the image at certain levels can be uniformly represented accordingly sized super pixel.
5. Instead of averaging the progress in the depth buffer take the min value.
6. Take care of uniform dark areas by taking the max value at gradient computation for the hoelder alpha (not too sure here what is meant).
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

#include "MitchellFilter.h"

#include "AdaptivePathTraceRenderContext.h"

#include "Scenes.h"

using namespace optix;

const char* const SAMPLE_NAME = "master-thesis";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

//AdaptivePathTraceContext* adaptive_PT_context;
Context        context = 0;
uint32_t       width = 512;
uint32_t       height = 512;
bool           use_pbo = true;

int            frame_number = 1;
int            sqrt_num_samples = sqrt(currentLevelAdaptiveSampleNumber);
int            num_samples = getInitialRenderNumSamples();
int            rr_begin_depth = 1;
Program        pgram_intersection = 0;
Program        pgram_bounding_box = 0;

// Camera state
TrackballCamera* camera;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

bool useAdaptivePostProcessing = false;

//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
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

Buffer getPerWindowVarianceBuffer() 
{
	return context["per_window_variance_buffer_input"]->getBuffer();
}

Buffer getPerRayBudgetBuffer()
{
	return context["additional_rays_buffer_input"]->getBuffer();
}

Buffer getPerRayWindowSizeBuffer()
{
	return context["window_size_buffer"]->getBuffer();
}

Buffer getOutputDepthBuffer() 
{
	return context["output_scene_depth_buffer"]->getBuffer();
}

// Debug
Buffer getDepthGradientBuffer()
{
	return context["depth_gradient_buffer"]->getBuffer();
}

// Debug
Buffer getHoelderAlphaBuffer()
{
	return context["hoelder_alpha_buffer"]->getBuffer();
}

// Debug
Buffer getHoelderAdaptiveSceneDepthBuffer() 
{
	return context["hoelder_adaptive_scene_depth_buffer"]->getBuffer();
}

// Debug
Buffer getHoelderRefinementBuffer()
{
	return context["hoelder_refinement_buffer"]->getBuffer();
}

// Debug
Buffer getTotalSampleCountBuffer()
{
	return context["total_sample_count_buffer"]->getBuffer();
}

// Debug
Buffer getPerWindowVarianceBufferOutput()
{
	return context["per_window_variance_buffer_output"]->getBuffer();
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
	Context context,
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

void loadComplexGeometry(Context context, const Scene scene)
{
	// set up material
	//Material diffuse = context->createMaterial();
	//const char *adaptive_ptx = sutil::getPtxString(SAMPLE_NAME, "adaptive.cu");
	const char *ptx = sutil::getPtxString(SAMPLE_NAME, "optixPathTracer.cu");
	Program diffuse_ch = context->createProgramFromPTXString(ptx, "diffuseTextured");
	Program diffuse_ah = context->createProgramFromPTXString(ptx, "shadow");
	Program diffuse_ah_radiance = context->createProgramFromPTXString(ptx, "any_hit_radiance");
	//diffuse->setClosestHitProgram(0, diffuse_ch);
	//diffuse->setAnyHitProgram(1, diffuse_ah);

	Material diffuse_light = context->createMaterial();
	Program diffuse_em = context->createProgramFromPTXString(ptx, "diffuseEmitter");
	diffuse_light->setClosestHitProgram(0, diffuse_em);

	// create geometry instances

	// load model
	Aabb model_aabb;
	OptiXMesh mesh;

	mesh.closest_hit = diffuse_ch;
	mesh.any_hit = diffuse_ah;

	mesh.has_any_hit_radiance = true;
	mesh.any_hit_radiance = diffuse_ah_radiance;

	mesh.context = context;

	Matrix4x4 scale_matrix;
	scale_matrix.setRow(0, make_float4(scene.scale.x, 0.0f,			 0.0f,		    0.0f));
	scale_matrix.setRow(1, make_float4(0.0f,		  scene.scale.y, 0.0f,		    0.0f));
	scale_matrix.setRow(2, make_float4(0.0f,		  0.0f,			 scene.scale.z, 0.0f));
	scale_matrix.setRow(3, make_float4(0.0f,		  0.0f,			 0.0f,		    1.0f));

	loadMesh(scene.file_name, mesh, scale_matrix);

	model_aabb.set(mesh.bbox_min, mesh.bbox_max);
	float a = model_aabb.maxExtent();
	context["far_plane"]->setFloat(model_aabb.maxExtent() * 2.0f);

	GeometryGroup geometry_group = context->createGeometryGroup();
	geometry_group->addChild(mesh.geom_instance);
	geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
	context["top_object"]->set(geometry_group);
	context["top_shadower"]->set(geometry_group);

	const float3 white = make_float3(0.8f, 0.8f, 0.8f);

	const float3 light_em = scene.parallelogram_lights[0].emission;

	ParallelogramLight light = scene.parallelogram_lights[0];
	float3 relativeCorner = light.corner;
	light.corner = make_float3(model_aabb.center().x + relativeCorner.x, model_aabb.center().y + relativeCorner.y, model_aabb.center().z + relativeCorner.z);

	Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(ParallelogramLight));
	light_buffer->setSize(1u);
	memcpy(light_buffer->map(), &light, sizeof(light));
	light_buffer->unmap();
	context["lights"]->setBuffer(light_buffer);

	GeometryInstance light_parallelogram = createParallelogram(context, 
		light.corner,
		light.v1,
		light.v2);
	setMaterial(light_parallelogram, diffuse_light, "emission_color", light_em);
}

void setupAdaptivePostprocessing(Context context, uint32_t width, uint32_t height)
{
	commandListAdaptive = context->createCommandList();

	commandListAdaptive->appendLaunch(1, width, height);
	commandListAdaptive->finalize();
	useAdaptivePostProcessing = true;
}

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
	camera->update(frame_number);
	//updateCurrentInitialRenderNumSamples(context, camera->getChanged(), maxAdditionalRaysPerRenderRun, num_samples);
	context->launch(0, width, height);

	if (useAdaptivePostProcessing)
	{		
		//setupVarianceBuffer();
		//if (isInitialRenderRunDone())
		//{
			updateCurrentLevelAdaptiveSampleCount(context, camera->getChanged());
			commandListAdaptive->execute();
		//}
		//sutil::displayBufferGL(getOutputDepthBuffer());
		//sutil::displayBufferGL(getDepthGradientBuffer());
		//sutil::displayBufferGL(getHoelderAlphaBuffer());
		//sutil::displayBufferGL(getHoelderRefinementBuffer());
		//sutil::displayBufferGL(getTotalSampleCountBuffer());
		//sutil::displayBufferGL(getHoelderAdaptiveSceneDepthBuffer());
		//sutil::displayBufferGL(getPerWindowVarianceBuffer());
	}
	sutil::displayBufferGL(getOutputBuffer());
	camera->setChanged(false);
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
		camera->setEye(camera->getEye() + (camera->getLookat() - camera->getEye()) * scale);
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

		camera->setRotation(arcball.rotate(b, a));
		camera->setChanged(true);
	}

	mouse_prev_pos = make_int2(x, y);
}

void glutResize(int w, int h)
{
	if (w == (int)width && h == (int)height) return;

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

	AdaptivePathTraceContext* adaptive_PT_context = new AdaptivePathTraceContext();

    try { 
		Scene scene = KilleroosSceneSetupAndGet();

		width = scene.width;
		height = scene.height;

		glutInitialize(&argc, argv);

#ifndef __APPLE__
		glewInit();
#endif
		//setCurrentInitialRenderNumSamples(num_samples);

		adaptive_PT_context->setWidth(width);
		adaptive_PT_context->setHeight(height);

		adaptive_PT_context->setFrameNumber(frame_number);
		adaptive_PT_context->setSqrtNumSample(sqrt_num_samples);
		adaptive_PT_context->setNumSample(num_samples);

		adaptive_PT_context->setWindowSize(windowSize);
		adaptive_PT_context->setMaxAdditionalRaysTotal(maxAdditionalRaysTotal);
		adaptive_PT_context->setMaxAdditionalRaysPerRenderRun(maxAdditionalRaysPerRenderRun);

		adaptive_PT_context->createAdaptiveContext(SAMPLE_NAME, "optixPathTracer.cu", "adaptiveOptixPathTracer.cu", use_pbo, rr_begin_depth);

		float2 radius = make_float2(1.0f, 1.0f);

		setupMitchellFilter(adaptive_PT_context->getContext()->getContext(), radius, sqrt_num_samples);

		camera = new TrackballCamera(adaptive_PT_context->getContext()->getContext(), (int)width, (int)height);
		camera->setup(scene.eye, scene.look_at, scene.up, scene.fov, true);

		loadComplexGeometry(adaptive_PT_context->getContext()->getContext(), scene);

		// Adaptive post processing setup
		setupAdaptivePostprocessing(adaptive_PT_context->getContext()->getContext(), width, height);

		context = adaptive_PT_context->getContext()->getContext();

		context->validate();

		if (out_file.empty())
		{
			glutRun();
		}
		else
		{
			context->launch(0, width, height);
			// Adaptive test execute if active
			if (useAdaptivePostProcessing)
			{
				commandListAdaptive->execute();
				sutil::displayBufferPPM(out_file.c_str(), getOutputBuffer(), false);
			}
			else
			{
				sutil::displayBufferPPM(out_file.c_str(), getOutputBuffer(), false);
			}
			//destroyContext();
		}

		delete adaptive_PT_context;
        return( 0 );

    } SUTIL_CATCH( adaptive_PT_context->getContext()->getContext()->get() )
}


void printUsageAndExit( const char* argv0 )
{
  fprintf( stderr, "Usage  : %s [options]\n", argv0 );
  fprintf( stderr, "Options: --file | -f <filename>      Specify file for image output\n" );
  fprintf( stderr, "         --help | -h                 Print this usage message\n" );
  fprintf( stderr, "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n" );
  exit(1);
}