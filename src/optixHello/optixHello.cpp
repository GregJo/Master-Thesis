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
		  I intend to write my adaptive implementation into seperate source files, using the OptiX xpp-wrapper as a base.
		- I probably have to add .png loading capability to the .obj loader used by the OptiX xpp-wrapper
		- Later on switch to and/or extend on the "optixPathTracer" project
	- device/.cu side: 
		- Implement it as an additional adaptive pass (must ensure that the first resulting image is completely avaible)
		- Have a user set maximum sample budget per pixel
		- Have a function that is providing current additional adaptive sample count, based on the neighborhood of the current launch index.
		- In case of race conditions use atomics
		- Launch addtional, adaptive rays with "rtTrace".

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
3. Find out how to implement multiple passes.
4. Find out how the progressive ray tracing example works.
5. Find out how the post processing framework works.
6. Evaluate, whether multiple passes, progressive or post prcessing framework approach is suited, for adaptive ray launching. (an additional pass seems so far most appropriate)
extra: Find out whether a "dynanmic" adaptive ray launching is possible, i.e. necessary neighborhood of 
the current 2D launch index in the ray generation program recieved the output values necessary for adaptive
ray launching and so the additional rays can be launched. For that i will need at least one more addition output buffer.
*/

#include <optix.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sutil.h>

#include "commonStructs.h"

#include <optixu/optixpp_namespace.h>

#include <OptiXMesh.h>


using namespace optix;

void printUsageAndExit( const char* argv0 );

// Tutorial predeclares
void cameraSetup();
void geometrySetup();

optix::float3 eye;
optix::float3 U, V, W;

void cameraSetup()
{
	// Tutorial Camera Setup
	optix::float3 lookAt, up;
	eye.x = 500.0f;
	eye.y = 1000.0f;
	eye.z = 0.0f;

	lookAt.x = 0.01f;
	lookAt.y = 0.01f;
	lookAt.z = 0.01f;

	up.x = 0.0f;
	up.y = 1.0f;
	up.z = 0.0f;

	sutil::calculateCameraVariables(eye, lookAt, up, 90.0f, 4.0f / 3.0f, U, V, W);
}

void setupLights()
{
	/*
	BasicLight lights[] = {
		{ make_float3(-5.0f, 60.0f, -16.0f), make_float3(1.0f, 1.0f, 1.0f), 1 }
	};

	RTbuffer light_buffer;
	RT_CHECK_ERROR( rtBufferCreate( context, &light_buffer ) );
	//Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	RT_CHECK_ERROR( rtBufferSetFormat( RT_FORMAT_USER ) );
	//light_buffer->setFormat(RT_FORMAT_USER);
	RT_CHECK_ERROR( rtBufferSetElementSize( sizeof(BasicLight) ) );
	//light_buffer->setElementSize(sizeof(BasicLight));
	RT_CHECK_ERROR( rtBufferSetSize1D( light_buffer, sizeof( lights ) / sizeof( lights[0] ) ) );
	//light_buffer->setSize(sizeof(lights) / sizeof(lights[0]));
	void* lightData;
	RT_CHECK_ERROR( rtBufferMap(light_buffer, &lightData) );
	memcpy(lightData, lights, sizeof(lights));
	rtBufferUnmap(light_buffer);
	//light_buffer->unmap();
	RTvariable lights_variable;
	RT_CHECK_ERROR( rtContextDeclareVariable(context, &lights_variable) );

	RT_CHECK_ERROR(rtVariableSetObject(lights_variable, light_buffer));
	//context["lights"]->set(light_buffer);
	*/
}

void setupAdditionalRaysBuffer() 
{

}

// Additional rays variables (make a struct if necessary)
//struct AddtionalAdaptiveRaysBuffer
//{
//	unsigned int maxPerLaunchIdxRayBudget;
//	unsigned int** perLaunchIdxRayBudgets;
//
//	AddtionalAdaptiveRaysBuffer(const unsigned int maxRayBudget, const unsigned int width, const unsigned int height)
//	{
//		maxPerLaunchIdxRayBudget = maxRayBudget;
//		for (unsigned int i = 0; i < width; i++)
//		{
//			for (unsigned int j = 0; j < width; j++)
//			{
//				perLaunchIdxRayBudgets[i][j] = maxPerLaunchIdxRayBudget+1;
//			}
//		}
//	}
//};

int main(int argc, char* argv[])
{
    RTcontext context = 0;

    try { 

        /* Primary RTAPI objects */
        RTprogram closest_hit_program;
		
		RTprogram pinhole_camera;
		RTprogram exception;
		RTprogram miss;

        RTbuffer  buffer;
		// Additional rays test buffer
		RTbuffer  additional_rays_buffer;

        /* Parameters */
        RTvariable result_buffer;
		// Additional rays test buffer
		RTvariable additional_rays_buffer_variable;
		RTvariable;
        RTvariable draw_color;

		RTvariable top_object;
		RTvariable top_shadower;
		RTvariable bg_color;
		RTvariable bad_color;

		RTvariable radiance_ray_type;
		RTvariable shadow_ray_type;

		RTvariable scene_epsylon;

		RTvariable eye_variable;
		RTvariable U_variable;
		RTvariable V_variable;
		RTvariable W_variable;

        char outfile[800];

        int width  = 800u;
        int height = 600u;
        int i;

        outfile[0] = '\0';

        sutil::initGlut( &argc, argv );

        for( i = 1; i < argc; ++i ) {
            if( strcmp( argv[i], "--help" ) == 0 || strcmp( argv[i], "-h" ) == 0 ) {
                printUsageAndExit( argv[0] );
            } else if( strcmp( argv[i], "--file" ) == 0 || strcmp( argv[i], "-f" ) == 0 ) {
                if( i < argc-1 ) {
                    strcpy( outfile, argv[++i] );
                } else {
                    printUsageAndExit( argv[0] );
                }
            } else if ( strncmp( argv[i], "--dim=", 6 ) == 0 ) {
                const char *dims_arg = &argv[i][6];
                sutil::parseDimensions( dims_arg, width, height );
            } else {
                fprintf( stderr, "Unknown option '%s'\n", argv[i] );
                printUsageAndExit( argv[0] );
            }
        }

        /* Create our objects and set state */
        RT_CHECK_ERROR( rtContextCreate( &context ) );
        RT_CHECK_ERROR( rtContextSetRayTypeCount( context, 2 ) );
        RT_CHECK_ERROR( rtContextSetEntryPointCount( context, 1 ) );

		// Enaple printing in .cu files
		rtContextSetPrintEnabled(context, 1);
		rtContextSetPrintBufferSize(context, 4096);

		// Tutorial begin
		const char *ptx = sutil::getPtxString("optixHello", "draw_color.cu");
		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "pinhole_camera", &pinhole_camera));
		RT_CHECK_ERROR(rtContextSetRayGenerationProgram(context, 0u, pinhole_camera));
		
		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "exception", &exception));
		RT_CHECK_ERROR(rtContextSetMissProgram(context, 0u, exception));

		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "miss", &miss));
		RT_CHECK_ERROR(rtContextSetMissProgram(context, 0u, miss));

		// Tutorial variable declaration
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "top_object", &top_object));
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "top_shadower", &top_shadower));
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "bg_color", &bg_color));
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "bad_color", &bad_color));
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "radiance_ray_type", &radiance_ray_type));
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "shadow_ray_type", &shadow_ray_type));
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "scene_epsylon", &scene_epsylon));

		RT_CHECK_ERROR(rtContextDeclareVariable(context, "eye", &eye_variable));
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "U", &U_variable));
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "V", &V_variable));
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "W", &W_variable));

		// Tutorial variable set
		RT_CHECK_ERROR(rtVariableSet3f(bg_color, 0.2f, 0.2f, 0.2f));
		RT_CHECK_ERROR(rtVariableSet3f(bad_color, 1.0f, 1.0f, 0.0f));
		RT_CHECK_ERROR(rtVariableSet1ui(radiance_ray_type, 0u));
		RT_CHECK_ERROR(rtVariableSet1ui(shadow_ray_type, 1u));
		RT_CHECK_ERROR(rtVariableSet1f(scene_epsylon, 1.e-4f));

		cameraSetup();

		RT_CHECK_ERROR(rtVariableSet3f(eye_variable, eye.x, eye.y, eye.z));
		RT_CHECK_ERROR(rtVariableSet3f(U_variable, U.x, U.y, U.z));
		RT_CHECK_ERROR(rtVariableSet3f(V_variable, V.x, V.y, V.z));
		RT_CHECK_ERROR(rtVariableSet3f(W_variable, W.x, W.y, W.z));

		// Tutorial geometry setup
		const std::string filePath("../bin/Data/sponza/sponza.obj");
		OptiXMesh mesh;
		
		RTgeometrygroup geometry_group;
		RTacceleration acceleration;

		RT_CHECK_ERROR(rtProgramCreateFromPTXString(context, ptx, "closest_hit_radiance0", &closest_hit_program));
		RT_CHECK_ERROR(rtContextSetRayGenerationProgram(context, 0u, pinhole_camera));

		mesh.context = Context::take(context);
		mesh.closest_hit = Program::take(closest_hit_program);
		loadMesh(filePath, mesh);

		RT_CHECK_ERROR(rtGeometryGroupCreate(context, &geometry_group));

		// IMPORTANT!!!: Use the Optix XPP wrapper for further proceeding
		unsigned int index;
		RT_CHECK_ERROR(rtGeometryGroupGetChildCount(geometry_group, &index));
		RT_CHECK_ERROR(rtGeometryGroupSetChildCount(geometry_group, index + 1));
		RT_CHECK_ERROR(rtGeometryGroupSetChild(geometry_group, index, mesh.geom_instance->get()));

		RT_CHECK_ERROR(rtAccelerationCreate(context, &acceleration));
		RT_CHECK_ERROR(rtAccelerationSetBuilder(acceleration, "Trbvh"));
		RT_CHECK_ERROR(rtGeometryGroupSetAcceleration(geometry_group, acceleration));

		
		RT_CHECK_ERROR( rtVariableSetObject(top_object, geometry_group) );
		RT_CHECK_ERROR(rtVariableSetObject(top_shadower, geometry_group));

		BasicLight lights[] = 
		{
			{ make_float3(-0.5f,  0.25f, -1.0f), make_float3(0.2f, 0.2f, 0.25f), 0, 0 },
			{ make_float3(-0.5f,  0.0f ,  1.0f), make_float3(0.1f, 0.1f, 0.10f), 0, 0 },
			{ make_float3(0.5f,  0.5f ,  0.5f), make_float3(0.7f, 0.7f, 0.65f), 1, 0 }
		};

		RTbuffer light_buffer;
		RT_CHECK_ERROR(rtBufferCreate(context, RT_BUFFER_INPUT, &light_buffer));
		RT_CHECK_ERROR(rtBufferSetFormat(light_buffer, RT_FORMAT_USER));
		RT_CHECK_ERROR(rtBufferSetElementSize(light_buffer, sizeof(BasicLight)));
		RT_CHECK_ERROR(rtBufferSetSize1D(light_buffer, sizeof(lights) / sizeof(lights[0])));
		void* lightData = nullptr;
		RT_CHECK_ERROR(rtBufferMap(light_buffer, &lightData));
		memcpy(lightData, lights, sizeof(lights));
		rtBufferUnmap(light_buffer);
		//light_buffer->unmap();
		RTvariable lights_variable;
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "lights", &lights_variable));

		RT_CHECK_ERROR(rtVariableSetObject(lights_variable, light_buffer));

		// Tutorial end!
        RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_OUTPUT, &buffer ) );
		RT_CHECK_ERROR(rtBufferSetFormat(buffer, RT_FORMAT_UNSIGNED_BYTE4));
        RT_CHECK_ERROR( rtBufferSetSize2D( buffer, width, height ) );
		RT_CHECK_ERROR(rtContextDeclareVariable(context, "output_buffer", &result_buffer));
        RT_CHECK_ERROR( rtVariableSetObject( result_buffer, buffer ) );

		// additional max ray budget and rays buffer
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
		RT_CHECK_ERROR(rtBufferCreate(context, RT_BUFFER_INPUT_OUTPUT, &additional_rays_buffer));
		RT_CHECK_ERROR(rtBufferSetFormat(additional_rays_buffer, RT_FORMAT_UNSIGNED_BYTE4));	// normally RT_FORMAT_UNSIGNED_BYTE would be enough, or RT_FORMAT_UNSIGNED_INT, 
																								// depending on the magnitude of additional samples one plans to send, 
																								// but i might want to visualize it
		RT_CHECK_ERROR(rtBufferSetSize2D(additional_rays_buffer, width, height));

		int a = sizeof(perLaunchIdxRayBudgets);

		void* additionalRaysData = nullptr;
		RT_CHECK_ERROR(rtBufferMap(additional_rays_buffer, &additionalRaysData));
		memcpy(additionalRaysData, perLaunchIdxRayBudgets, sizeof(unsigned char) * width * height * 4);
		rtBufferUnmap(additional_rays_buffer);

		RT_CHECK_ERROR(rtContextDeclareVariable(context, "additional_rays_buffer", &additional_rays_buffer_variable));
		RT_CHECK_ERROR(rtVariableSetObject(additional_rays_buffer_variable, additional_rays_buffer));

		RT_CHECK_ERROR( rtProgramDeclareVariable( pinhole_camera, "draw_color", &draw_color ) );
		RT_CHECK_ERROR( rtVariableSet3f( draw_color, 0.462f, 0.725f, 0.0f ) );

        /* Run */
        RT_CHECK_ERROR( rtContextValidate( context ) );
        RT_CHECK_ERROR( rtContextLaunch2D( context, 0 /* entry point */, width, height ) );

        /* Display image */
        if( strlen( outfile ) == 0 ) {
            sutil::displayBufferGlut( argv[0], buffer );
        } else {
            sutil::displayBufferPPM( outfile, buffer, false);
        }

        /* Clean up */
        RT_CHECK_ERROR( rtBufferDestroy( buffer ) );
		RT_CHECK_ERROR( rtProgramDestroy( pinhole_camera ) );
		RT_CHECK_ERROR( rtProgramDestroy( closest_hit_program ) );
		RT_CHECK_ERROR( rtProgramDestroy( miss ) );
		RT_CHECK_ERROR( rtProgramDestroy( exception) );
        RT_CHECK_ERROR( rtContextDestroy( context ) );

		delete[] perLaunchIdxRayBudgets;

        return( 0 );

    } SUTIL_CATCH( context )
}


void printUsageAndExit( const char* argv0 )
{
  fprintf( stderr, "Usage  : %s [options]\n", argv0 );
  fprintf( stderr, "Options: --file | -f <filename>      Specify file for image output\n" );
  fprintf( stderr, "         --help | -h                 Print this usage message\n" );
  fprintf( stderr, "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n" );
  exit(1);
}


