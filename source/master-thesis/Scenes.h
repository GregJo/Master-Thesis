#pragma once
#include "Scene.h"
#include "optixPathTracer.h"
#include "../sutil/sampleConfig.h"

uint maxAdaptiveLevel = 10;

Scene BMW6SceneSetupAndGet() 
{
	Scene bmw6;
	bmw6.file_name = std::string(DATA_DIR).append("/bmw-m6/tyrant_monkey_bmw249.obj");
	bmw6.scale = make_float3(6.0f);
	bmw6.eye = make_float3(-11.0f, 0.8f, 5.0f) * bmw6.scale;
	bmw6.look_at = make_float3(-2.0f, -0.5f, 0.0f) * bmw6.scale;
	bmw6.up = make_float3(0.0f, 1.0f, 0.0f);
	/*bmw6.fov = 30.0f;*/
	bmw6.fov = 45.0f;

	ParallelogramLight light;
	light.corner = make_float3(0.0f, 100.0f, 50.0f);
	light.v1 = make_float3(200.0f, 0.0f, 0.0f);
	light.v2 = make_float3(0.0f, 0.0f, -200.0f);
	light.normal = normalize(cross(light.v1, light.v2));
	light.emission = make_float3(5.0f, 5.0f, 2.0f);

	bmw6.parallelogram_lights.push_back(light);

	//bmw6.width = 700;
	//bmw6.height = 500;
	bmw6.width = pow(2, maxAdaptiveLevel);
	bmw6.height = pow(2, maxAdaptiveLevel);

	return bmw6;
}


Scene BarcelonaPavillonSetupAndGet()
{
	Scene barcelona_pavillon;
	barcelona_pavillon.file_name = "../bin/Data/barcelona-pavilion/pavillon_barcelone_v1.2.obj";
	barcelona_pavillon.scale = make_float3(-1.0f, -1.0f, 1.0f);
	barcelona_pavillon.eye = make_float3(-15.0f, 2.25f, 15.0f) * barcelona_pavillon.scale;
	barcelona_pavillon.look_at = make_float3(7.0f, 1.75f, -3.0f) * barcelona_pavillon.scale;
	barcelona_pavillon.up = normalize(make_float3(0.0f, 1.0f, 0.0f) * barcelona_pavillon.scale);
	barcelona_pavillon.fov = 45.0f;

	ParallelogramLight light;
	light.corner = make_float3(5.0f, -200.0f, -5.0f);
	light.v1 = make_float3(-200.0f, 0.0f, 0.0f);
	light.v2 = make_float3(0.0f, 0.0f, 200.0f);
	//light.corner = make_float3(50, -100.0f, -100.0f);
	//light.v1 = make_float3(0.0f, 200.0f, 0.0f);
	//light.v2 = make_float3(0.0f, 0.0f, 200.0f);
	light.normal = -normalize(cross(light.v1, light.v2));
	light.emission = make_float3(15.0f, 15.0f, 5.0f);

	barcelona_pavillon.parallelogram_lights.push_back(light);

	//barcelona_pavillon.width = 800;
	//barcelona_pavillon.height = 425;
	barcelona_pavillon.width = pow(2, maxAdaptiveLevel);
	barcelona_pavillon.height = pow(2, maxAdaptiveLevel);

	return barcelona_pavillon;
}


Scene CrytekSponzaSceneSetupAndGet()
{
	// make_float3(-500.0f, 1250.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), 35.0f
	Scene crytek_sponza;
	crytek_sponza.file_name = "../bin/Data/sponza/sponza.obj";
	crytek_sponza.scale = make_float3(1.0f);
	crytek_sponza.eye = make_float3(-500.0f, 1250.0f, 0.0f) * crytek_sponza.scale;
	crytek_sponza.look_at = make_float3(0.0f, 0.0f, 0.0f) * crytek_sponza.scale;
	crytek_sponza.up = make_float3(0.0f, 1.0f, 0.0f);
	crytek_sponza.fov = 35.0f;

	ParallelogramLight light;
	light.corner = make_float3(0.0f, 100.0f, 50.0f);
	light.v1 = make_float3(200.0f, 0.0f, 0.0f);
	light.v2 = make_float3(0.0f, 0.0f, -200.0f);
	light.normal = normalize(cross(light.v1, light.v2));
	light.emission = make_float3(15.0f, 15.0f, 5.0f);

	crytek_sponza.parallelogram_lights.push_back(light);

	crytek_sponza.width = pow(2, maxAdaptiveLevel);
	crytek_sponza.height = pow(2, maxAdaptiveLevel);

	return crytek_sponza;
}

Scene KilleroosSceneSetupAndGet()
{
	// make_float3(-500.0f, 1250.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), 35.0f
	Scene killeroos;
	killeroos.file_name = std::string(DATA_DIR).append("/Models/killeroos/killeroos_scene.obj");
	killeroos.scale = make_float3(1.0f);
	killeroos.eye = make_float3(-100.0f, 750.0f, -1000.0f) * killeroos.scale;
	killeroos.look_at = make_float3(-100.0f, -250.0f, 1000.0f) * killeroos.scale;
	killeroos.up = make_float3(0.0f, 1.0f, 0.0f);
	killeroos.fov = 39.0f;

	ParallelogramLight light;
	light.corner = make_float3(150.0f, 250.0f, 50.0f);
	light.v1 = make_float3(30.0f, 0.0f, 0.0f);
	light.v2 = make_float3(0.0f, 0.0f, 30.0f);
	light.normal = -normalize(cross(light.v1, light.v2));
	light.emission = make_float3(200.0f, 200.0f, 200.0f);

	killeroos.parallelogram_lights.push_back(light);

	killeroos.width = pow(2, maxAdaptiveLevel);
	killeroos.height = pow(2, maxAdaptiveLevel);

	return killeroos;
}