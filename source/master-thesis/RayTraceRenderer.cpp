#include "RayTraceRenderContext.h"

//
// Member function definitions
//

RayTraceRenderContext::RayTraceRenderContext(const char* const SAMPLE_NAME, const uint32_t width, const uint32_t height) : m_project_name(SAMPLE_NAME)
{
	m_width = width;
	m_height = height;
}

void RayTraceRenderContext::setupContext(int usage_report_level, void* logger, const bool use_pbo)
{
	// Set up context
	m_context = Context::create();
	m_context->setRayTypeCount(2);
	m_context->setEntryPointCount(1);
	if (usage_report_level > 0)
	{
		m_context->setUsageReportCallback(usageReportCallback, usage_report_level, logger);
	}

	m_context["radiance_ray_type"]->setUint(0u);
	m_context["shadow_ray_type"]->setUint(1u);
	m_context["scene_epsilon"]->setFloat(1.e-6f);

	Buffer buffer = sutil::createOutputBuffer(m_context, RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height, use_pbo);
	m_context["output_buffer"]->set(buffer);

	// Ray generation program
	const char *ptx = sutil::getPtxString(m_project_name.c_str(), "draw_color.cu");
	Program ray_gen_program = m_context->createProgramFromPTXString(ptx, "pinhole_camera");
	m_context->setRayGenerationProgram(0, ray_gen_program);

	// Exception program
	Program exception_program = m_context->createProgramFromPTXString(ptx, "exception");
	m_context->setExceptionProgram(0, exception_program);
	m_context["bad_color"]->setFloat(1.0f, 0.0f, 1.0f);

	// Miss program
	m_context->setMissProgram(0, m_context->createProgramFromPTXString(sutil::getPtxString(m_project_name.c_str(), "draw_color.cu"), "miss"));
	m_context["bg_color"]->setFloat(0.2f, 0.2f, 0.2f);

	m_context->declareVariable("input_buffer")->set(getOutputBuffer());
}

void RayTraceRenderContext::setupScene(float3 camera_eye, float3 camera_up, float3 camera_lookat, BasicLight* lights, const unsigned int numberOfLights, const std::string& filename)
{
	Aabb model_aabb;

	// load and setup mesh
	loadAndSetupMesh(model_aabb, filename);

	// setup camera
	setupCamera(model_aabb, camera_eye, camera_up, camera_lookat);

	// setup lights
	setupLights(model_aabb, lights, numberOfLights);
}

void RayTraceRenderContext::display(const char* outfile, const char* option)
{
	/* Run */
	m_context->validate();
	m_context->launch(0, m_width, m_height);

	/* Display image */
	if (strlen(outfile) == 0) {
		sutil::displayBufferGlut(option, getOutputBuffer());
	}
	else {
		sutil::displayBufferPPM(outfile, getOutputBuffer(), false);
	}
}

//
// Helper functions
//
void RayTraceRenderContext::setupCamera(Aabb model_aabb, float3 camera_eye, float3 camera_up, float3 camera_lookat)
{
	const float max_dim = fmaxf(model_aabb.extent(0), model_aabb.extent(1)); // max of x, y components

	camera_eye = model_aabb.center() + make_float3(0.0f, 0.0f, max_dim*1.5f);
	camera_lookat = model_aabb.center();
	camera_up = make_float3(0.0f, 1.0f, 0.0f);

	const float vfov = 90.0f;
	const float aspect_ratio = static_cast<float>(m_width) /
		static_cast<float>(m_height);

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

	float3 U;
	float3 V;
	float3 W;
	sutil::calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
		U, V, W, true);

	m_context["eye"]->setFloat(camera_eye);
	m_context["U"]->setFloat(U);
	m_context["V"]->setFloat(V);
	m_context["W"]->setFloat(W);
}

void RayTraceRenderContext::setupLights(const Aabb model_aabb, BasicLight* lights, const unsigned int numberOfLights)
{
	const float max_dim = fmaxf(model_aabb.extent(0), model_aabb.extent(1)); // max of x, y components

	for (uint i = 0; i < numberOfLights; i++)
	{
		lights[i].pos *= max_dim * 10.0f;
	}

	Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(BasicLight));
	light_buffer->setSize(numberOfLights);
	memcpy(light_buffer->map(), lights, numberOfLights * sizeof(BasicLight));
	light_buffer->unmap();

	m_context["lights"]->set(light_buffer);
}

void RayTraceRenderContext::loadAndSetupMesh(Aabb &model_aabb, const std::string& filename)
{
	OptiXMesh mesh;
	mesh.context = m_context;
	loadMesh(filename, mesh);

	model_aabb.set(mesh.bbox_min, mesh.bbox_max);

	GeometryGroup geometry_group = m_context->createGeometryGroup();
	geometry_group->addChild(mesh.geom_instance);
	geometry_group->setAcceleration(m_context->createAcceleration("Trbvh"));
	m_context["top_object"]->set(geometry_group);
	m_context["top_shadower"]->set(geometry_group);
}

void RayTraceRenderContext::destroyContext()
{
	if (m_context)
	{
		m_context->destroy();
		m_context = 0;
	}
}