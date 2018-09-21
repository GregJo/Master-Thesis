#include <optixu/optixpp_namespace.h>

using namespace optix;

//
// Mitchell Filter
//

class MitchellFilter 
{
public:
	MitchellFilter(float2 radius, int filterTableWidth, float B) : _radius(radius), _B(B), _C(0.5f * (1.0f - _B)), _filterTableWidth(filterTableWidth)
	{
		_filterTable = new float[_filterTableWidth * _filterTableWidth];
		_invRadius.x = 1.0f / radius.x;
		_invRadius.y = 1.0f / radius.y;
	}
	MitchellFilter(float2 radius, int filterTableWidth, float B, float C) : _radius(radius), _B(B), _C(C), _filterTableWidth(filterTableWidth)
	{
		_filterTable = new float[_filterTableWidth * _filterTableWidth];
		_invRadius.x = 1.0f / radius.x;
		_invRadius.y = 1.0f / radius.y;
	}

	~MitchellFilter() 
	{
		delete[] _filterTable;
	}

	float Mitchell1D(float x) {
		x = std::abs(2 * x);
		if (x > 1)
			return ((-_B - 6 * _C) * x*x*x + (6 * _B + 30 * _C) * x*x +
			(-12 * _B - 48 * _C)*x + (8 * _B + 24 * _C)) * (1.f / 6.f);
		else
			return ((12 - 9 * _B - 6 * _C) * x*x*x +
			(-18 + 12 * _B + 6 * _C) * x*x +
				(6 - 2 * _B)) * (1.f / 6.f);
	}

	float Evaluate(const float2 &p) {
		return Mitchell1D(p.x * _invRadius.x) * Mitchell1D(p.y * _invRadius.y);
	}

	void fillFilterTable()
	{
		float2 p;
		p.x = -_radius.x;
		p.y = -_radius.y;

		float radiusXDelta = _radius.x / _filterTableWidth;
		float radiusYDelta = _radius.y / _filterTableWidth;

		float2 originalP;
		originalP.x = p.x;
		originalP.y = p.y;

		for (size_t i = 0; i < _filterTableWidth * _filterTableWidth; i++)
		{
			_filterTable[i] = Evaluate(p);

			p.x = originalP.x + radiusXDelta * 2.0f * ((i+1) % _filterTableWidth);
			p.y = originalP.y + radiusYDelta * 2.0f * ((i+1) / _filterTableWidth);
		}
	}

	float* getFilterTable() { return _filterTable; }

	int getFilterTableWidth() { return _filterTableWidth; }

	float2 getRadius() { return _radius; }
	float2 getInvRadius() { return _invRadius; }

private:
	const float2 _radius;							/// Radius size in pixel
	float2 _invRadius;
	const float _B = 0.5f, _C = 0.5 * (1.0f - _B);
	const int _filterTableWidth = 16;
	float* _filterTable;
};

void setupMitchellFilter(Context context, float2 radius, int sqrt_num_samples)//int num_samples)
{
	int expected_samples_count = sqrt_num_samples * sqrt_num_samples;//num_samples;// 5;

	MitchellFilter mitchellFilter(radius, 16, 0.5f);
	mitchellFilter.fillFilterTable();

	printf("Mitchell filter values!\n\n");
	int idx = 0;
	for (size_t i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			printf("[ %.3f ]", mitchellFilter.getFilterTable()[idx]);
			idx++;
		}
		printf("\n");
	}

	// Additional rays test buffer setup
	Buffer mitchell_filter_table = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT, mitchellFilter.getFilterTableWidth(), mitchellFilter.getFilterTableWidth(), false);
	memcpy(mitchell_filter_table->map(), mitchellFilter.getFilterTable(), sizeof(float) * mitchellFilter.getFilterTableWidth() * mitchellFilter.getFilterTableWidth());
	mitchell_filter_table->unmap();

	context["mitchell_filter_table"]->set(mitchell_filter_table);
	context["mitchell_filter_radius"]->setFloat(mitchellFilter.getRadius());
	context["mitchell_filter_inv_radius"]->setFloat(mitchellFilter.getInvRadius());
	context["mitchell_filter_table_width"]->setInt(mitchellFilter.getFilterTableWidth());
	context["expected_samples_count"]->setInt(expected_samples_count);
}