#pragma once

#include "PathTraceRenderContext.h"
#include "TrackballCamera.h"
#include <sutil.h>

class AdaptivePathTraceContext : public PathTraceRenderContext
{
public:
	
	AdaptivePathTraceContext() : PathTraceRenderContext() {}
	~AdaptivePathTraceContext() {}

	void createAdaptiveContext(const char* const SAMPLE_NAME, const char* fileNameOptixPrograms, const char* fileNameAdaptiveOptixPrograms, const bool use_pbo, int rr_begin_depth)
	{
		createContext(SAMPLE_NAME, fileNameOptixPrograms, use_pbo, rr_begin_depth);

		_setupPerRayTotalBudgetBuffer();
		_setupPerRayWindowSizeBuffer();

		const char *adaptive_ptx = sutil::getPtxString(SAMPLE_NAME, fileNameAdaptiveOptixPrograms);

		Buffer output_filter_sum_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["output_filter_sum_buffer"]->set(output_filter_sum_buffer);
		
		Buffer output_filter_x_sample_sum_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["output_filter_x_sample_sum_buffer"]->set(output_filter_x_sample_sum_buffer);
		
		_context->declareVariable("adaptive_samples_budget_buffer")->set(_context["additional_rays_buffer_input"]->getBuffer());

		Buffer hoelder_refinement_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["hoelder_refinement_buffer"]->set(hoelder_refinement_buffer);

		Buffer total_sample_count_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["total_sample_count_buffer"]->set(total_sample_count_buffer);

		Buffer hoelder_adaptive_scene_depth_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["hoelder_adaptive_scene_depth_buffer"]->set(hoelder_adaptive_scene_depth_buffer);

		Program adaptive_ray_gen_program = _context->createProgramFromPTXString(adaptive_ptx, "pathtrace_camera_adaptive");
		_context->setRayGenerationProgram(1, adaptive_ray_gen_program);

		Buffer output_scene_depth_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["output_scene_depth_buffer"]->set(output_scene_depth_buffer);

		// This buffer is for debug
		Buffer depth_gradient_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["depth_gradient_buffer"]->set(depth_gradient_buffer);

		// This buffer is for debug
		Buffer hoelder_alpha_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["hoelder_alpha_buffer"]->set(hoelder_alpha_buffer);

		_context["window_size"]->setUint(_windowSize);
		_context["max_ray_budget_total"]->setUint(_maxAdditionalRaysTotal);
		_context["max_per_frame_samples_budget"]->setUint(_maxAdditionalRaysPerRenderRun);
	}

	void setWindowSize(unsigned int windowSize)
	{
		_windowSize = windowSize;
	}

	void setMaxAdditionalRaysTotal(unsigned int maxAdditionalRaysTotal)
	{
		_maxAdditionalRaysTotal = maxAdditionalRaysTotal;
	}

	void setMaxAdditionalRaysPerRenderRun(unsigned int maxAdditionalRaysPerRenderRun)
	{
		_maxAdditionalRaysPerRenderRun = maxAdditionalRaysPerRenderRun;
	}

private:
	uint _windowSize; 
	uint _maxAdditionalRaysTotal;
	uint _maxAdditionalRaysPerRenderRun;

private:

	void _setupPerRayTotalBudgetBuffer()
	{
		int* perPerRayBudget = new int[_width * _height * 4];

		// initialize additional rays buffer
		for (unsigned int i = 0; i < _width * _height; i++)
		{
			perPerRayBudget[i * 4] = static_cast<unsigned int>(_maxAdditionalRaysTotal);
			perPerRayBudget[i * 4 + 1] = static_cast<unsigned int>(_maxAdditionalRaysTotal);
			perPerRayBudget[i * 4 + 2] = static_cast<unsigned int>(_maxAdditionalRaysTotal);
			perPerRayBudget[i * 4 + 3] = static_cast<unsigned int>(_maxAdditionalRaysTotal);
		}

		// Additional rays test buffer setup
		Buffer additional_rays_buffer = sutil::createInputOutputBuffer(_context, RT_FORMAT_UNSIGNED_INT4, _width, _height, false);
		memcpy(additional_rays_buffer->map(), perPerRayBudget, sizeof(int) * _width * _height * 4);
		additional_rays_buffer->unmap();
		_context["additional_rays_buffer_input"]->set(additional_rays_buffer);

		delete[] perPerRayBudget;
	}

	void _setupPerRayWindowSizeBuffer()
	{
		int* perPerRayWindow_Size = new int[_width * _height * 4];

		// initialize additional rays buffer
		for (unsigned int i = 0; i < _width * _height; i++)
		{
			perPerRayWindow_Size[i * 4] = static_cast<unsigned int>(_windowSize);
			perPerRayWindow_Size[i * 4 + 1] = static_cast<unsigned int>(_windowSize);
			perPerRayWindow_Size[i * 4 + 2] = static_cast<unsigned int>(_windowSize);
			perPerRayWindow_Size[i * 4 + 3] = static_cast<unsigned int>(_windowSize);
		}

		// Additional rays test buffer setup
		Buffer additional_rays_buffer = sutil::createInputOutputBuffer(_context, RT_FORMAT_UNSIGNED_INT4, _width, _height, false);
		memcpy(additional_rays_buffer->map(), perPerRayWindow_Size, sizeof(int) * _width * _height * 4);
		additional_rays_buffer->unmap();
		_context["window_size_buffer"]->set(additional_rays_buffer);

		delete[] perPerRayWindow_Size;
	}
};