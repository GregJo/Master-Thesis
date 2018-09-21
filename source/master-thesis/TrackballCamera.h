#pragma once
#include "Camera.h"
#include <Arcball.h>

class TrackballCamera : public Camera
{
public:
	TrackballCamera() : Camera() {}
	TrackballCamera(Context &context, const unsigned int width, const unsigned int height) : Camera(context, width, height), m_rotate_sens(1.0f), m_changed(false) {}
	~TrackballCamera() {}

	void update(int &frame_number)
	{
		m_calculateContextVariables();

		const Matrix4x4 frame = Matrix4x4::fromBasis(
			normalize(m_camera_u),
			normalize(m_camera_v),
			normalize(-m_camera_w),
			m_camera_lookat);
		const Matrix4x4 frame_inv = frame.inverse();
		// Apply camera rotation twice to match old SDK behavior
		const Matrix4x4 trans = frame*m_camera_rotate*m_camera_rotate*frame_inv;

		m_camera_eye = make_float3(trans*make_float4(m_camera_eye, 1.0f));
		m_camera_lookat = make_float3(trans*make_float4(m_camera_lookat, 1.0f));
		m_camera_up = make_float3(trans*make_float4(m_camera_up, 0.0f));

		m_calculateContextVariables();

		m_camera_rotate = Matrix4x4::identity();

		if (m_changed) // reset accumulation
			frame_number = 1;
		//m_changed = false;
		//m_context["camera_changed"]->setInt(0);

		m_setContextVariables(frame_number);
	}

	void setRotateSens(const float rotate_sens) { m_rotate_sens = rotate_sens; }

	void setChanged(const bool changed) 
	{ 
		m_changed = changed; 
		m_context["camera_changed"]->setInt(changed);
	}

	bool getChanged()
	{
		return m_changed;
	}

	void setRotation(Matrix4x4 camera_rotate) { m_camera_rotate = camera_rotate; }

	Matrix4x4 getRotation() { return m_camera_rotate; }

private:

	float	m_rotate_sens;

	bool	m_changed;

	sutil::Arcball m_arcball;

	void m_setContextVariables(int &frame_number)
	{
		m_context["frame_number"]->setUint(frame_number++);
		m_context["eye"]->setFloat(m_camera_eye);
		m_context["U"]->setFloat(m_camera_u);
		m_context["V"]->setFloat(m_camera_v);
		m_context["W"]->setFloat(m_camera_w);
	}

	void m_calculateContextVariables()
	{
		sutil::calculateCameraVariables(
			m_camera_eye, m_camera_lookat, m_camera_up, m_fov, m_aspect_ratio,
			m_camera_u, m_camera_v, m_camera_w, true);
	}

	void m_calculateRotation()
	{
		const Matrix4x4 frame = Matrix4x4::fromBasis(
			normalize(m_camera_u),
			normalize(m_camera_v),
			normalize(-m_camera_w),
			m_camera_lookat);
		const Matrix4x4 frame_inv = frame.inverse();
		// Apply camera rotation twice to match old SDK behavior
		const Matrix4x4 m_rotation = frame*m_camera_rotate*m_camera_rotate*frame_inv;

		m_camera_rotate = Matrix4x4::identity();
	}
};