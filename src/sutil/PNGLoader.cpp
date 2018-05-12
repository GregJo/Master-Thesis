#include "PNGLoader.h"
#include <optixu/optixu_math_namespace.h>
#include <iostream>

using namespace optix;
using namespace lodepng;
//  
//  PNGLoader class definition
//

PNGLoader::PNGLoader(const std::string& filename, const LodePNGColorType PNGcolorType) : m_raster(nullptr), m_nx(0), m_ny(0)
{
	if (filename.empty()) return;

	unsigned int channels = -1;
	switch (PNGcolorType)
	{
	case LCT_GREY:
		channels = 1;
		break;
	case LCT_RGB:
		channels = 3;
		break;
	case LCT_PALETTE:
		channels = 1;
		break;
	case LCT_GREY_ALPHA:
		channels = 2;
		break;
	case LCT_RGBA:
		channels = 4;
		break;
	}

	//
	// Following two values for now hardcoded, to esure an easy data copy implementation between host and device buffer in the "loadTexture( ... )" function
	//
	channels = 3;
	LodePNGColorType PNGcolorType2 = LCT_RGB;

	std::vector<unsigned char> out;

	//decode(out, m_nx, m_ny, filename);

	State state; //optionally customize this one

	unsigned error = decode(out, m_nx, m_ny, filename, PNGcolorType2);
	if (!error)
	{
		m_raster = new unsigned char[m_nx * m_ny * channels];
		memcpy(m_raster, out.data(), m_nx * m_ny * channels);
	}
	else
	{
		std::cerr << "PNGLoader( '" << filename << "' ) failed to load!" << std::endl;
		std::cerr << "Error Message: " << lodepng_error_text(error) << std::endl;
	}
}

PNGLoader::~PNGLoader()
{
	delete[] m_raster;
	m_raster = 0;
}

SUTILAPI bool PNGLoader::failed() const
{
	return false;
}

unsigned int PNGLoader::width() const
{
	return m_nx;
}


unsigned int PNGLoader::height() const
{
	return m_ny;
}

unsigned char* PNGLoader::raster() const
{
	return m_raster;
}

SUTILAPI optix::TextureSampler PNGLoader::loadTexture(optix::Context context,
	const optix::float3& default_color)
{
	// Create tex sampler and populate with default values
	optix::TextureSampler sampler = context->createTextureSampler();
	sampler->setWrapMode(0, RT_WRAP_REPEAT);
	sampler->setWrapMode(1, RT_WRAP_REPEAT);
	sampler->setWrapMode(2, RT_WRAP_REPEAT);
	sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
	sampler->setMaxAnisotropy(1.0f);
	sampler->setMipLevelCount(1u);
	sampler->setArraySize(1u);

	if (failed()) {

		// Create buffer with single texel set to default_color
		optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, 1u, 1u);
		unsigned char* buffer_data = static_cast<unsigned char*>(buffer->map());
		buffer_data[0] = (unsigned char)clamp((int)(default_color.x * 255.0f), 0, 255);
		buffer_data[1] = (unsigned char)clamp((int)(default_color.y * 255.0f), 0, 255);
		buffer_data[2] = (unsigned char)clamp((int)(default_color.z * 255.0f), 0, 255);
		buffer_data[3] = 255;
		buffer->unmap();

		sampler->setBuffer(0u, 0u, buffer);
		// Although it would be possible to use nearest filtering here, we chose linear
		// to be consistent with the textures that have been loaded from a file. This
		// allows OptiX to perform some optimizations.
		sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

		return sampler;
	}

	const unsigned int nx = width();
	const unsigned int ny = height();

	// Create buffer and populate with PNG data
	optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, nx, ny);
	unsigned char* buffer_data = static_cast<unsigned char*>(buffer->map());

	for (unsigned int i = 0; i < nx; ++i) {
		for (unsigned int j = 0; j < ny; ++j) {

			unsigned int ppm_index = ((ny - j - 1)*nx + i) * 3;
			unsigned int buf_index = ((j)*nx + i) * 4;

			buffer_data[buf_index + 0] = raster()[ppm_index + 0];
			buffer_data[buf_index + 1] = raster()[ppm_index + 1];
			buffer_data[buf_index + 2] = raster()[ppm_index + 2];
			buffer_data[buf_index + 3] = 255;
			//buffer_data[buf_index + 3] = raster()[ppm_index + 3];
		}
	}

	buffer->unmap();

	sampler->setBuffer(0u, 0u, buffer);
	sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

	return sampler;
}

//  
//  Utility functions 
//

optix::TextureSampler loadPNGTexture(optix::Context context,
	const std::string& filename,
	const optix::float3& default_color,
	const LodePNGColorType PNGColorType)
{
	PNGLoader png(filename, PNGColorType);
	return png.loadTexture(context, default_color);
}