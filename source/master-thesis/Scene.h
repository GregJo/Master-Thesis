#pragma once

#include "optixPathTracer.h"
#include "Camera.h"

#include <string>
#include <vector>

struct Scene
{
	std::vector<ParallelogramLight> parallelogram_lights;
	optix::float3 eye;
	optix::float3 look_at;
	optix::float3 up;
	float fov;
	optix::float3 scale;
	std::string file_name;

	uint32_t width;
	uint32_t height;
};
