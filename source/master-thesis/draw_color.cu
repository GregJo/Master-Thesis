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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////   Adaptive Additional Rays Test   //////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

rtDeclareVariable(float3,		// type of the variable used in an RT_PROGRAM in this .cu file
	shading_normal,				// name of the variable used in an RT_PROGRAM in this .cu file
	attribute shading_normal,	// semantic variable declared on the API in the corresponding .cpp file, attribute is to specify that the variable is part of a struct
	);

// per ray data struct
struct PerRayData_radiance
{
	float3 result;				// struct variable carrying our calculated output
	float  importance;
	int depth;
	int done;
};

rtDeclareVariable(PerRayData_radiance,
	prd_radiance, 
	rtPayload,							//This is a semantic name, not an API declared variable name to bind user data to
	);

RT_PROGRAM void closest_hit_radiance0()
{
	prd_radiance.result = normalize(rtTransformNormal(	// transforms n as a normal using the current active transformation stack (the inverse transpose)
		RT_OBJECT_TO_WORLD,								// other option would be RT_WORLD_TO_OBJECT
		shading_normal))
		*0.5f + 0.5f;
	prd_radiance.done = true;
}

rtDeclareVariable(float3, bg_color, , );

// Miss program for a ray, in case a ray misses the geometry give it the background color.
RT_PROGRAM void miss()
{
	prd_radiance.result = bg_color;
	prd_radiance.done = true;
}

static __device__ __inline__ uchar4 make_color(const float3& c)
{
	return make_uchar4(static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
	static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
	static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
	255u);                                                 /* A */
};

static __device__ __inline__ float3 revert_color(const uchar4& c)
{
	return make_float3(static_cast<unsigned char>(__saturatef(c.z)*1.0f/255.99f),  /* B */
		static_cast<unsigned char>(__saturatef(c.y)*1.0f/255.99f),  /* G */
		static_cast<unsigned char>(__saturatef(c.x)*1.0f/255.99f)  /* R */);                                                 /* A */
};

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, bad_color, , );
rtBuffer<uchar4, 2>   output_buffer;

/*
For post processing create multiple ray genereation programs.
The first ray generation program provides the input image.
The other ray generation programs do custom work on the input data initially provided by the first  

Useful comment from NVIDIA guy on their dev talk forum:
"There always has always been an easy path to custom post-processing within optix -- your own ray-gen programs which do post-processing, as you mention, 
or your own CUDA kernels. 
The postprocessing allows you to add optix launches to the pipeline (either for rendering or custom postprocess operations) so that you can use the pipeline 
as your all-in-one per-frame render pipeline."
*/
RT_PROGRAM void pinhole_camera()
{
	size_t2 screen = output_buffer.size();

	float2 d = make_float2(launch_index) /
		make_float2(screen) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	Ray ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon);
	PerRayData_radiance prd;
	prd.importance = 1.f;
	prd.depth = 0;
	prd.done = false;

	rtTrace(top_object, ray, prd); /* find out when its done, its important to know whether the code proceeds after this line after "rtTrace" is 'finished',
								   or if it starts a parallel subroutine and the code advances without waiting for "rtTrace" to finish (i assume the latter,
								   due to what i read in the technical overview -> the former is true, evidence by testing) */

	output_buffer[launch_index] = make_color(prd.result);
}

///*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
///* Adaptive additional rays variables */
//rtDeclareVariable(uint, max_per_launch_idx_ray_budget, , ) = static_cast<uint>(5u);		/* this variable will be written by the user */
//rtBuffer<uchar4, 2>   additional_rays_buffer;											/* this buffer will be initialized by the host, but must also be modified by the graphics device */
//
//rtBuffer<uchar4, 2>   input_buffer;														/* this buffer contains the initially rendered picture to be post processed */
//rtBuffer<uchar4, 2>   post_process_output_buffer;										/* this buffer contains the result, processed with additional adaptive rays */
//
//rtDeclareVariable(float, window_size, , );
//
//static __device__ __inline__ float compute_window_variance(uint2 center, uint window_size)
//{
//	float mean = 0.f;
//	float variance = 0.f;
//	uint squared_window_size = window_size * window_size;
//	uint2 upper_top_left_window = make_uint2(center.x - static_cast<uint>(static_cast<float>(window_size) / 2.f), center.y - static_cast<uint>(static_cast<float>(window_size) / 2.f));
//	/* compute mean value */
//	for (uint i = 0; i < squared_window_size; i++)
//	{
//		uint2 idx = make_uint2(static_cast<uint>(i / window_size) + upper_top_left_window.x, static_cast<uint>(i % window_size) + upper_top_left_window.y);
//		float3 input_buffer_val = revert_color(input_buffer[idx]);
//		mean += 1.f/3.f * (input_buffer_val.x + input_buffer_val.y + input_buffer_val.z);
//	}
//
//	mean *= 1.f/ squared_window_size;
//
//	/* compute variance */
//	for (uint i = 0; i < squared_window_size; i++)
//	{
//		uint2 idx = make_uint2(static_cast<uint>(i / window_size) + upper_top_left_window.x, static_cast<uint>(i % window_size) + upper_top_left_window.y);
//		float3 input_buffer_val = revert_color(input_buffer[idx]);
//		float var = 1.f / 3.f * (input_buffer_val.x + input_buffer_val.y + input_buffer_val.z);
//		variance += var;
//	}
//
//	variance = 1.f / squared_window_size * (variance) - (mean * mean);
//
//	return variance;
//};
//
//static __device__ __inline__ uint compute_variance_based_additional_samples_number(uint window_size) 
//{
//	uint additional_samples_number = 0;
//	/* check if box window is in buffer window */
//	/* actually compute 'additional_samples_number' */
//	return additional_samples_number;
//};
//
//RT_PROGRAM void adaptive_camera()
//{
//	/* Testing for additional adaptive rays. Added jittering for test purposes. */
//	
//	/* 
//		Postpone launching additional rays until first currently traced ray output is avaible (extend to neighborhood after success).
//			- 1. Postponing will be done with a loop, which will run indefinitely and does nothing (maybe use observer pattern here, more elegant than having a loop with an if statement), 
//				 until a condition is met, in this case when the output buffer has been written (-> no longer necessary, because the code advances after "rtTrace" only after its done).
//			  2. Upon reaching the written output buffer state which i will modify the additional "additional_rays_buffer" values, which are initialized with ("max_per_launch_idx_ray_budget" + 1)
//			     so that they contain an arbitary smaller or value (but only corresponding (neighboring) values to the current launchIdx).
//			  3. After setting the current additional(, adaptive) ray budget i break/leave the loop and start another, that launches another loop, in which i launch additional rays,
//			     according to the current budget and add/write the results into the output buffer.
//		Additional adaptive rays count will be avaible in the "additional_rays_buffer"
//	*/
//	size_t2 screen = post_process_output_buffer.size();
//
//	float2 d = make_float2(launch_index) /
//		make_float2(screen) * 2.f - 1.f;
//
//	uint additional_rays_count = static_cast<uint>(additional_rays_buffer[launch_index].x);
//
//	float3 ray_origin = eye;
//	float3 ray_direction = normalize(d.x*U + d.y*V + W);
//
//	/* Make the following 'adaptive pass' test to a real adaptive pass (for that i must ensure, that the first resulting image is completely avaible). */
//	//if (prd.done)
//	//{
//		additional_rays_count = static_cast<uint>(input_buffer[launch_index].x) % (max_per_launch_idx_ray_budget + 1u);
//		//rtPrintf("Launch index: %u, %u; Additional rays count: %u !\n\n", launch_index.x, launch_index.y, additional_rays_count);
//		float jitter = static_cast<float>(additional_rays_count) / static_cast<float>(max_per_launch_idx_ray_budget);
//		float jitterScale = 0.1f;
//		jitter = jitter * jitterScale;
//
//		if (additional_rays_count <= 0)
//		{
//			post_process_output_buffer[launch_index] = make_color(bad_color);
//		}
//
//		while (additional_rays_count > 0u)
//		{
//			//rtPrintf("Additional rays left: %u !\n", additional_rays_count);
//			float3 jittered_ray_origin;
//
//			jittered_ray_origin.x = ray_origin.x + jitter;
//			jittered_ray_origin.y = ray_origin.y - jitter;
//			jittered_ray_origin.z = ray_origin.z + jitter;
//
//			float3 jittered_ray_direction;
//
//			jittered_ray_direction.x = ray_direction.x + jitter;
//			jittered_ray_direction.y = ray_direction.y - jitter;
//			jittered_ray_direction.z = ray_direction.z + jitter;
//
//			Ray ray2(jittered_ray_origin, jittered_ray_direction, radiance_ray_type, scene_epsilon);
//			PerRayData_radiance prd2;
//			prd2.importance = 1.f;
//			prd2.depth = 0;
//			prd2.done = false;
//
//			rtTrace(top_object, ray2, prd2);
//
//			/*post_process_output_buffer[launch_index] = make_color(revert_color(input_buffer[launch_index]) + prd2.result);*/
//			post_process_output_buffer[launch_index] = make_color(make_float3(1.0f));
//			additional_rays_count--;
//
//			jitter = static_cast<float>(additional_rays_count) / static_cast<float>(max_per_launch_idx_ray_budget);
//			jitterScale = jitterScale * -1.f;
//			jitter = jitter * jitterScale;
//		}
//	//}
//}

/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

RT_PROGRAM void exception()
{
	output_buffer[launch_index] = make_color(bad_color);
}