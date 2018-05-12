
#pragma once

#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include <string>

#include <lodepng.h>

#include <iostream>
#include <fstream>
//
// Utility Functions
//

SUTILAPI optix::TextureSampler loadPNGTexture(optix::Context context,
											const std::string& png_filename,
											const optix::float3& default_color,
											const LodePNGColorType PNGColorType = LCT_RGBA);

//
// PNGLoader class declaration 
//

class PNGLoader 
{
public:
	PNGLoader(const std::string& filename, const LodePNGColorType PNGcolorType = LCT_RGBA);
	~PNGLoader();

	SUTILAPI optix::TextureSampler loadTexture(optix::Context context,
											const optix::float3& default_color );

	SUTILAPI bool           failed() const;
	SUTILAPI unsigned int   width() const;
	SUTILAPI unsigned int   height() const;
	SUTILAPI unsigned char* raster() const;

private:
	unsigned int   m_nx;
	unsigned int   m_ny;
	unsigned char* m_raster;

	unsigned int m_channels;
};

//
// Helper functions
//

std::vector<unsigned char> readFile(const std::string& filename);