/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu2.hpp>

#include "support.h"

unsigned char average_kernel(int ox, int oy, size_t stride, const unsigned char *m, size_t elemPerPx)
{
	float scaling = 1.0 / ((ox/elemPerPx*2+1)*(oy*2+1));
	float res = 0;
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx)
			res += m[y*(int)stride+x];
	return res * scaling;
}

unsigned char average_kernel_1d(int o, size_t stride, const unsigned char *m, size_t elemPerPx)
{
    float res = 0;
    int num_elems = 0;
    int step = 1;
    if (stride == 1) {
        step = elemPerPx;
    }
    for (int i = -o*step; i <= o*step; i += step) {
        res += m[i*(int)stride];
        num_elems++;
    }
    float scaling = 1/((float)num_elems);
    return res*scaling;
}



unsigned char gaussian_kernel(int o, size_t stride, const unsigned char *m, const skepu2::Vec<float> stencil, size_t elemPerPx)
{
    float res = 0;
    int step = 1;
    if (stride == 1) {
        step = elemPerPx;
    }
    for (int i = -o; i <= o; ++i) {
        res += m[i*(int)(stride*step)] * stencil[i];
    }
    return res;
}




int main(int argc, char* argv[])
{
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " input output radius [backend]\n";
		exit(1);
	}
	
	LodePNGColorType colorType = LCT_RGB;
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[4])};
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFile = outputFileName + ss.str();
	
	// Read the padded image into a matrix. Create the output matrix without padding.
	// Padded version for 2D MapOverlap, non-padded for 1D MapOverlap
	ImageInfo imageInfo;
	skepu2::Matrix<unsigned char> inputMatrixPad = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu2::Matrix<unsigned char> inputMatrix = ReadPngFileToMatrix(inputFileName, colorType, imageInfo);
	skepu2::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	// more containers...?
	
	// Original version
	{
		auto conv = skepu2::MapOverlap(average_kernel);
		conv.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
		conv.setBackend(spec);
	
		auto timeTaken = skepu2::benchmark::measureExecTime([&]
		{
			conv(outputMatrix, inputMatrixPad, imageInfo.elementsPerPixel);
		});
	
		WritePngFileMatrix(outputMatrix, outputFile + "-average.png", colorType, imageInfo);
		std::cout << "Time for combined: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	// Separable version
	// use conv.setOverlapMode(skepu2::Overlap::[ColWise RowWise]);
	// and conv.setOverlap(<integer>)
	{
		auto conv = skepu2::MapOverlap(average_kernel_1d);
        conv.setOverlapMode(skepu2::Overlap::ColRowWise);
        conv.setOverlap(radius);
		conv.setBackend(spec);
	
		auto timeTaken = skepu2::benchmark::measureExecTime([&]
		{
			conv(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
		});
		
	    WritePngFileMatrix(outputMatrix, outputFile + "-separable.png", colorType, imageInfo);
		std::cout << "Time for separable: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	// Separable gaussian
	{
		skepu2::Vector<float> stencil = sampleGaussian(radius);
		auto conv = skepu2::MapOverlap(gaussian_kernel);
        conv.setOverlapMode(skepu2::Overlap::ColRowWise);
        conv.setOverlap(radius);
		conv.setBackend(spec);
			
		// skeleton instance, etc here (remember to set backend)
	
		auto timeTaken = skepu2::benchmark::measureExecTime([&]
		{
			conv(outputMatrix, inputMatrix, stencil, imageInfo.elementsPerPixel);
		});
	
	    WritePngFileMatrix(outputMatrix, outputFile + "-gaussian.png", colorType, imageInfo);
		std::cout << "Time for gaussian: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	
	return 0;
}


