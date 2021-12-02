
#include "LayerMaxPooling.h"
#include <assert.h>

CLayerMaxPooling2D::CLayerMaxPooling2D()
{
}

CLayerMaxPooling2D::~CLayerMaxPooling2D()
{
}

bool	CLayerMaxPooling2D::Setup(	size_t inputFeatureCount, size_t inputSizeX, size_t inputSizeY,
									size_t poolSizeX, size_t poolSizeY,
									size_t padding, size_t stride)
{
	if (stride == 0)
		stride = 1;
	if (inputSizeX % stride != 0 ||
		inputSizeY % stride != 0)
	{
		fprintf(stderr, "Input size should be a multiple of stride");
		assert(false);
		return false;
	}

	m_ConvParams.m_KernelCount = inputFeatureCount;
	m_ConvParams.m_KernelSizeX = poolSizeX;
	m_ConvParams.m_KernelSizeY = poolSizeY;
	m_ConvParams.m_KernelStride = stride;
	m_ConvParams.m_InputPadding = padding;
	m_ConvParams.m_InputImageCount = inputFeatureCount;
	m_ConvParams.m_InputSizeX = inputSizeX;
	m_ConvParams.m_InputSizeY = inputSizeY;
	return true;
}

void	CLayerMaxPooling2D::FeedForward(const float *input, size_t rangeMin, size_t rangeMax)
{
}

void	CLayerMaxPooling2D::BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax)
{
}

void	CLayerMaxPooling2D::BackPropagateError(const float *prevOutput, const CLayer* nextLayer, size_t rangeMin, size_t rangeMax)
{
}

void	CLayerMaxPooling2D::UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax)
{
}

void	CLayerMaxPooling2D::GatherSlopes(float *dst, size_t rangeMin, size_t rangeMax) const
{

}

size_t	CLayerMaxPooling2D::GetThreadingHint() const
{
	// TODO: Should be splittable in tasks
	return 1;
}

size_t	CLayerMaxPooling2D::GetDomainSize() const
{
	// TODO: Should be splittable in tasks
	return 1;
}

void	CLayerMaxPooling2D::AccumWeightsAndBiasDerivative(const float *prevOutput, size_t rangeMin, size_t rangeMax)
{

}
