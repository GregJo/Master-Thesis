#pragma once
#include "IOptixRenderContext.h"
#include "commonStructs.h"
#include <OptiXMesh.h>
#include <sutil.h>

using namespace optix;

struct UsageReporterLogger;
void usageReportCallback(int lvl, const char* tag, const char* msg, void* cbdata);

class RayTraceRenderContext: virtual public IOptixRenderContext
{
public:
	RayTraceRenderContext(const char* const SAMPLE_NAME, const uint32_t width, const uint32_t height);
	~RayTraceRenderContext() { destroyContext(); }

	void setupContext(int usage_report_level, void* logger, const bool use_pbo);

	void setupScene(float3 camera_eye, float3 camera_up, float3 camera_lookat, BasicLight* lights, const unsigned int numberOfLights, const std::string& filename);

	void display(const char* outfile, const char* option);
private:

	// Helper functions
	Buffer getOutputBuffer() { return m_context["output_buffer"]->getBuffer(); }

	void setupCamera(Aabb model_aabb, float3 camera_eye, float3 camera_up, float3 camera_lookat);

	void setupLights(const Aabb model_aabb, BasicLight* lights, const unsigned int numberOfLights);

	void loadAndSetupMesh(Aabb &model_aabb, const std::string& filename);

	void destroyContext();

	const std::string m_project_name;
};