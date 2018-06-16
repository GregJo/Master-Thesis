#pragma once
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

//#include <Arcball.h>
#include <sutil.h>

using namespace optix;

class Camera
{
public:
	Camera() {}
	Camera(Context &context, const unsigned int width, const unsigned int height) : m_context(context), m_width(width), m_height(height),
		m_camera_rotate(Matrix4x4::identity()),
		m_aspect_ratio(static_cast<float>(width) / static_cast<float>(height)) {}

	~Camera() {}

	void setup(const float3 camera_eye, const float3 camera_lookat, const float3 camera_up, const float fov, bool fov_is_vertical)
	{
		m_camera_eye = camera_eye;
		m_camera_lookat = camera_lookat;
		m_camera_up = camera_up;
		// right vector
		m_camera_right = normalize(cross(m_camera_lookat, m_camera_up));

		m_fov = fov;
		m_fov_is_vertical = fov_is_vertical;
	}

	void setCameraPosition(const float3 camera_eye) { m_camera_eye = camera_eye; }

	void setRotation(const Matrix4x4 rotation) { m_camera_rotate = rotation; }
	Matrix4x4 getRotation() { return m_camera_rotate; }

	void setUp(float3 up) { m_camera_up = up; }
	void setLookat(float3 lookat) { m_camera_lookat = lookat; }
	void setEye(float3 eye) { m_camera_eye = eye; }

	float3 getUp() { return m_camera_up; }
	float3 getLookat() { return m_camera_lookat; }
	float3 getEye() { return m_camera_eye; }

	unsigned int getWidth() { return m_width; }
	unsigned int getHeight() { return m_width; }

protected:
	Context			m_context;

	unsigned int	m_width;
	unsigned int	m_height;

	float3			m_camera_up;
	float3			m_camera_right;
	float3			m_camera_lookat;
	float3			m_camera_eye;

	float3			m_camera_u; 
	float3			m_camera_v;
	float3			m_camera_w;

	float			m_fov;
	float			m_aspect_ratio;
	bool			m_fov_is_vertical;

	Matrix4x4		m_camera_rotate;
	//Matrix4x4		m_rotation;

	//sutil::Arcball m_arcball;
};