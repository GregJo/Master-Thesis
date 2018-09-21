#pragma once

#include "PathTraceRenderContext.h"
#include "TrackballCamera.h"
#include "Scenes.h"
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

		// This buffer is for debug
		Buffer depth_gradient_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["depth_gradient_buffer"]->set(depth_gradient_buffer);

		// This buffer is for debug
		Buffer hoelder_alpha_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["hoelder_alpha_buffer"]->set(hoelder_alpha_buffer);

		_setupPerRayTotalBudgetBuffer();
		_setupPerRayWindowSizeBuffer();

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

		//_______________________________________________________________________________________________________________________________
		// Currently testing
		Buffer hoelder_level_output_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["hoelder_level_output_buffer"]->set(hoelder_level_output_buffer);
		
		Buffer hoelder_adaptive_level_scene_depth_buffer = sutil::createOutputBuffer(_context, RT_FORMAT_FLOAT4, _width, _height, use_pbo);
		_context["hoelder_adaptive_level_scene_depth_buffer"]->set(hoelder_adaptive_level_scene_depth_buffer);
		//_______________________________________________________________________________________________________________________________

		Program adaptive_ray_gen_program = _context->createProgramFromPTXString(adaptive_ptx, "pathtrace_camera_adaptive");
		_context->setRayGenerationProgram(1, adaptive_ray_gen_program);

		// Adaptive variables
		_context["window_size"]->setUint(_windowSize);
		_context["initial_window_size"]->setUint(_windowSize);
		//context["max_ray_budget_total"]->setUint(maxAdditionalRaysTotal);
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

	uint getMaxAdditionalRaysPerRenderRun() 
	{
		return _maxAdditionalRaysPerRenderRun;
	}

private:
	uint _windowSize; 
	uint _maxAdditionalRaysTotal;
	uint _maxAdditionalRaysPerRenderRun;
	
	uint _currentLevelAdaptiveSampleCount = 1;
	uint _currentAdaptiveLevel = 1;

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


uint firstAdaptiveLevel = 0;
uint adaptiveLevels = maxAdaptiveLevel - firstAdaptiveLevel;

CommandList commandListAdaptive;

// Variance based adaptive sampling specific
const uint windowSize = std::powf(2, adaptiveLevels);						// Powers of two are your friend.
												//const uint maxAdditionalRaysTotal = 50;
const uint maxAdditionalRaysTotal = 0;			// If using hoelder set this to zero. 
const uint maxAdditionalRaysPerRenderRun = 2;//std::powf(2, maxAdaptiveLevel);	// Powers of two are not only your friend, but a MUST here!
//float* perWindowVariance = nullptr;
//int* perPerRayBudget = nullptr;


uint currentLevelAdaptiveSampleNumber = std::powf(2, firstAdaptiveLevel + 1);
uint currentAdaptiveLevel = 1;

uint waitFramesNumber = currentLevelAdaptiveSampleNumber / maxAdditionalRaysPerRenderRun;

void resetAdaptiveLevelVariables()
{
	currentLevelAdaptiveSampleNumber = std::powf(2, firstAdaptiveLevel + 1);
	currentAdaptiveLevel = 1;
}

void updateCurrentLevelAdaptiveSampleCount(Context context, bool cameraChanged)
{
	if (cameraChanged)
	{
		resetAdaptiveLevelVariables();
	}
	if (waitFramesNumber > 0)
	{
		waitFramesNumber--;
	}
	if (currentAdaptiveLevel < adaptiveLevels && waitFramesNumber == 0)
	{
		currentAdaptiveLevel++;
		currentLevelAdaptiveSampleNumber *= 2;
		waitFramesNumber = currentLevelAdaptiveSampleNumber / maxAdditionalRaysPerRenderRun;
		printf("\n______________________________________________________________________________________________________\n\n");
		printf("Current level adaptive sample number (host): %u\n", currentLevelAdaptiveSampleNumber);
		printf("\n______________________________________________________________________________________________________\n\n");
	}
	context["current_level_adaptive_sample_count"]->setUint(currentLevelAdaptiveSampleNumber);
}

int getInitialRenderNumSamples()
{
	int numSamples = 0;

	for (int i = 0; i <= firstAdaptiveLevel; i++)
	{
		numSamples += pow(2,i);
	}

	return numSamples;
}

//void setupcurrentMaxAdaptiveLevelSamplesNumber(Context context, int width, int height, uint currentLevelSamplesNumber)
//{
//	int* currentMaxAdaptiveLevelSamplesNumber = new int[width * height * 4];
//
//	// initialize additional rays buffer
//	for (unsigned int i = 0; i < width * height; i++)
//	{
//		currentMaxAdaptiveLevelSamplesNumber[i * 4] = static_cast<unsigned int>(currentLevelSamplesNumber);
//		currentMaxAdaptiveLevelSamplesNumber[i * 4 + 1] = static_cast<unsigned int>(currentLevelSamplesNumber);
//		currentMaxAdaptiveLevelSamplesNumber[i * 4 + 2] = static_cast<unsigned int>(currentLevelSamplesNumber);
//		currentMaxAdaptiveLevelSamplesNumber[i * 4 + 3] = static_cast<unsigned int>(currentLevelSamplesNumber);
//	}
//
//	// Additional rays test buffer setup
//	Buffer current_max_adaptive_level_samples_number = sutil::createInputOutputBuffer(context, RT_FORMAT_UNSIGNED_INT4, width, height, false);
//	memcpy(current_max_adaptive_level_samples_number->map(), currentMaxAdaptiveLevelSamplesNumber, sizeof(int) * width * height * 4);
//	current_max_adaptive_level_samples_number->unmap();
//	context["current_max_adaptive_level_samples_number"]->set(current_max_adaptive_level_samples_number);
//
//	delete[] currentMaxAdaptiveLevelSamplesNumber;
//}