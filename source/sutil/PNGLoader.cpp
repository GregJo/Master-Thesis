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
	size_t suffix_pos = filename.find_last_of(".");
	std::string filename_suffix = filename.substr(suffix_pos + 1);
	std::cout << "Filename: " << filename << std::endl;
	//std::cout << "Filename suffix: " << filename_suffix << std::endl;
	if (filename_suffix != "png") return;

	std::vector<unsigned char> out;

	std::vector<unsigned char> raw_data = readFile(filename);

	lodepng::State state; //optionally customize this one

	unsigned error = lodepng::decode(out, m_nx, m_ny, state, raw_data);
	//std::cout << "Size of out vector: " << out.size() << std::endl;

	unsigned int m_nx2 = 0, m_ny2 = 0;
	std::vector<unsigned char> out2;
	unsigned error2 = lodepng::decode(out2, m_nx2, m_ny2, filename, LodePNGColorType::LCT_RGB);
	//std::cout << "Size of correct out vector: " << out.size() << std::endl << std::endl;

	// Print first 3 color values (9 floats all in all)
	//printf("First 3 colors of out vector:  [%u,%u,%u] | [%u,%u,%u] | [%u,%u,%u]\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8]);
	//printf("First 3 colors of out2 vector: [%u,%u,%u] | [%u,%u,%u] | [%u,%u,%u]\n\n", out2[0], out2[1], out2[2], out2[3], out2[4], out2[5], out2[6], out2[7], out2[8]);

	if (!error)
	{
		m_channels = -1;
		switch (state.info_png.color.colortype)
		{
		case LCT_GREY:
			m_channels = 1;
			break;
		case LCT_RGB:
			m_channels = 3;
			break;
		case LCT_PALETTE:
			m_channels = 1;
			break;
		case LCT_GREY_ALPHA:
			m_channels = 2;
			break;
		case LCT_RGBA:
			m_channels = 4;
			break;
		}
		//m_channels = 4;
		//m_raster = new unsigned char[m_nx * m_ny * m_channels];
		//memcpy(m_raster, out.data(), m_nx * m_ny * m_channels);

		m_raster = new unsigned char[out.size()];
		memcpy(m_raster, out.data(), out.size());

		//m_channels = 3;
		//m_raster = new unsigned char[m_nx2 * m_ny2 * m_channels];
		//memcpy(m_raster, out2.data(), m_nx2 * m_ny2 * m_channels);

		//std::cout << "Size from width, height and channels: " << m_nx * m_ny * m_channels << std::endl;
		//std::cout << "Channels: " << m_channels << std::endl << std::endl;
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

	unsigned char r, g, b, a = 0;
	for (unsigned int i = 0; i < nx; ++i) {
		for (unsigned int j = 0; j < ny; ++j) {

			unsigned int png_index = ((ny - j - 1)*nx + i) * 4;//m_channels;
			unsigned int buf_index = ((j)*nx + i) * 4;

			switch (m_channels) 
			{
				case 1:
					r = raster()[png_index + 0];
					g = raster()[png_index + 0];
					b = raster()[png_index + 0];
					a = 255;
					break;
				case 2: 
					r = raster()[png_index + 0];
					g = raster()[png_index + 1];
					b = raster()[png_index + 1];
					a = 255;
					break;
				case 3:
					r = raster()[png_index + 0];
					g = raster()[png_index + 1];
					b = raster()[png_index + 2];
					a = 255;
					break;
				case 4:
					r = raster()[png_index + 0];
					g = raster()[png_index + 1];
					b = raster()[png_index + 2];
					a = raster()[png_index + 3];
					break;
				case -1: break;
			}
			
			buffer_data[buf_index + 0] = r;
			buffer_data[buf_index + 1] = g;
			buffer_data[buf_index + 2] = b;
			buffer_data[buf_index + 3] = a;
		}
	}

	//printf("First color:  [ %u , %u , %u , %u ]\n", raster()[0], raster()[1], raster()[2], unsigned char(255));
	//printf("Second color:  [ %u , %u , %u , %u ]\n", raster()[3], raster()[4], raster()[5], unsigned char(255));
	//printf("Last [ R , G , B , A ]:  [ %u , %u , %u , %u ]\n\n\n", r, g, b, a);

	buffer->unmap();

	sampler->setBuffer(0u, 0u, buffer);
	sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

	return sampler;
}

//  
//  Utility functions definitions
//

optix::TextureSampler loadPNGTexture(optix::Context context,
	const std::string& filename,
	const optix::float3& default_color,
	const LodePNGColorType PNGColorType)
{
	PNGLoader png(filename, PNGColorType);
	return png.loadTexture(context, default_color);
}

//
// Helper functions definitions
//

std::vector<unsigned char> readFile(const std::string& filename)
{
	// Init
	std::ifstream* pFile = new std::ifstream();

	bool fileOpenSuccess = true;
	// Open
	pFile->open(filename, std::ios::in | std::ios::binary);
	if (!pFile->is_open())
	{
		fileOpenSuccess = false;
	}

	if (fileOpenSuccess)
	{
		// File size in bytes
		// get current position
		std::streampos curr = pFile->tellg();

		std::streampos begin = pFile->tellg();
		pFile->seekg(0, std::ios::end);
		std::streampos end = pFile->tellg();

		// restore position
		pFile->seekg(curr, std::ios::beg);
		int sizeInBytes = (size_t)(end - begin) + 1;

		char* dataFromDisk = new char[sizeInBytes];
		dataFromDisk[sizeInBytes - 1] = '\0';

		unsigned int readBytes = 0;
		char buffer[1024];

		int offset = 0;
		pFile->seekg(offset, std::ios::beg);

		for (size_t i = 0; !pFile->eof(); i += readBytes)
		{
			pFile->read(buffer, 1024);
			readBytes = (size_t)pFile->gcount();
			memcpy(dataFromDisk + i, buffer, readBytes);
		}

		pFile->close();

		std::vector<unsigned char> raw_data(dataFromDisk, dataFromDisk + sizeInBytes);

		if (dataFromDisk != nullptr)
		{
			delete[] dataFromDisk;
			dataFromDisk = nullptr;
		}

		return raw_data;
	}
	else
	{
		std::cerr << "Could not open File '%s'!" << filename.c_str() << std::endl;
		return std::vector<unsigned char>();
	}
}