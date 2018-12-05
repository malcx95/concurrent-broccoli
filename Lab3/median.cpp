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


void swap(unsigned char* x1, unsigned char* x2) {
    unsigned char tmp = *x1;
    *x1 = *x2;
    *x2 = tmp;
}


void insertion_sort(int length, unsigned char* array) {
    for (unsigned i = 1; i < length; ++i) {
        unsigned j = i;
        while (j > 0 && array[j-1] > array[j]) {
            swap(&array[j-1], &array[j]);
            j--;
        }
    }
}


unsigned char median_kernel(int ox, int oy, size_t stride, const unsigned char *image, size_t elemPerPx)
{
    int num_elements = (ox/elemPerPx*2+1)*(oy*2+1);
    unsigned char temp[num_elements];

    // copy to temp array
    unsigned index = 0;
	for (int y = -oy; y <= oy; ++y) {
		for (int x = -ox; x <= ox; x += elemPerPx) {
            temp[index] = image[y*(int)stride+x];
            index++;
        }
    }

    insertion_sort(num_elements, temp);

    return temp[(unsigned)num_elements/2];

}



int main(int argc, char* argv[])
{
	LodePNGColorType colorType = LCT_RGB;
	
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << "input output radius [backend]\n";
		exit(1);
	}
	
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[4])};
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + "-median.png";
		
	// Read the padded image into a matrix. Create the output matrix without padding.
	ImageInfo imageInfo;
	skepu2::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu2::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	
	// Skeleton instance
	auto calculateMedian = skepu2::MapOverlap(median_kernel);
	calculateMedian.setBackend(spec);
	calculateMedian.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
	
	auto timeTaken = skepu2::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
	});

	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);
	
	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";
	
	return 0;
}


