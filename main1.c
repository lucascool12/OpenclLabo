/* 
	Pretrained VGG16 convolutional neural network in C language
	GitHUB Page: https://github.com/ZFTurbo/VGG16-Pretrained-C
	Author: ZFTurbo
	
	Compilation: gcc -O3 -fopenmp -lm ZFC_VGG16_CPU.c -o ZFC_VGG16_CPU.exe
	Usage: ZFC_VGG16_CPU.exe <weights_path> <file_with_list_of_images> <output file> <output convolution features (optional)>
	Example: ZFC_VGG16_CPU.exe "weights.txt" "image_list.txt" "results.txt" 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "time_utils.h"
#include "imagenet_labels.h"

#define SIZE 224
#define NUM_CHANNELS 3
#define CONV_SIZE 3
#define KERNEL_SIZE 3
#define NUM_LAYERS 13
#define NUM_DENSE 3


#define IMAGE_INDEX(l, x, y)\
	((l) * (SIZE) * (SIZE) + (y) * (SIZE) + (x))	
#define PLANE_INDEX(l)\
    ((l) * (SIZE) * (SIZE))
#define MEM_BLOCK_DEPTH 512


#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "time_utils.h"
#include "ocl_utils.h"



// Weights and image block START
float *image;
int cshape[13][4] = { 
	{ 64, 3, CONV_SIZE, CONV_SIZE },
	{ 64, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE }
};
float *wc[NUM_LAYERS];
float *bc[NUM_LAYERS];
int dshape[3][2] = {
	{ 25088, 4096 },
	{ 4096, 4096 },
	{ 4096, 1000 }
};
float *wd[NUM_DENSE];
float *bd[NUM_DENSE];


// Blocks for intermediate convolutions
int mem_block_shape[3] = {MEM_BLOCK_DEPTH, SIZE, SIZE};

float *mem_block1;
float *mem_block2;
// Blocks for dense flatten layers
int mem_block_dense_shape = { 512 * 7 * 7 };
float *mem_block1_dense;
float *mem_block2_dense;

// Weights and image block END

// Create device buffers.
cl_mem dev_zeropad;

cl_mem dev_matrix;
	
cl_mem dev_weights;

cl_mem dev_out;

cl_kernel kernelAddbs;

cl_kernel kernelZeroPad;

cl_kernel kernel_inner_3x3;

int level;

void reset_mem_block(float *mem) {
	memset(mem, 0, MEM_BLOCK_DEPTH * SIZE * SIZE * sizeof(float));
}


void reset_mem_block_dense(float *mem) {
	int i;
	for (i = 0; i < mem_block_dense_shape; i++) {
		mem[i] = 0.0;
	}
}


void init_memory() {
	int l;

	// Init image memory
    image = (float *)malloc(NUM_CHANNELS * SIZE * SIZE * sizeof(float));

	// Init convolution weights
	for (l = 0; l < 13; l++) {
		wc[l] = malloc(cshape[l][0] * cshape[l][1] * cshape[l][2] * cshape[l][3] * sizeof(float));
		bc[l] = malloc(cshape[l][0] * sizeof(float));
	}

	// Init dense weights
	for (l = 0; l < 3; l++) {
		wd[l] = malloc(dshape[l][0] * dshape[l][1] * sizeof(float));
		bd[l] = malloc(dshape[l][1] * sizeof(float));
	}

	// Init mem_blocks
	mem_block1 = malloc(MEM_BLOCK_DEPTH * SIZE * SIZE * sizeof(float));
	mem_block2 = malloc(MEM_BLOCK_DEPTH * SIZE * SIZE * sizeof(float));
	reset_mem_block(mem_block1);
	reset_mem_block(mem_block2);

	// float zeropad[SIZE + 2][SIZE + 2][512] = {{{0.}}};

	cl_int error;
	dev_zeropad = clCreateBuffer(g_context, CL_MEM_READ_WRITE ,
				sizeof(cl_float) * (SIZE + 2)*(SIZE + 2)*512, NULL, &error);
	ocl_err(error);


	dev_matrix = clCreateBuffer(g_context, CL_MEM_READ_ONLY,
				MEM_BLOCK_DEPTH * SIZE * SIZE * sizeof(cl_float), NULL, &error);
	ocl_err(error);
		// Create device buffers.
	dev_weights = clCreateBuffer(g_context, CL_MEM_READ_ONLY,
			cshape[12][0] * cshape[12][1] * cshape[12][2] * cshape[12][3] * sizeof(float), NULL, &error);
	ocl_err(error);

	dev_out = clCreateBuffer(g_context, CL_MEM_READ_WRITE,
			MEM_BLOCK_DEPTH * SIZE * SIZE * sizeof(cl_float), NULL, &error);
	ocl_err(error);

	kernelAddbs = clCreateKernel(g_program, "add_bias_and_relu", &error);
	ocl_err(error);
	
	kernel_inner_3x3 = clCreateKernel(g_program, "inner_convolutie3x3", &error);
	ocl_err(error);

	kernelZeroPad = clCreateKernel(g_program, "zeropadConv", &error);
	ocl_err(error);
}


void free_memory() {
	int l;

	// Free image memory
    free(image);

	// Free convolution weights
	for (l = 0; l < 13; l++) {
		free(wc[l]);
		free(bc[l]);
	}

	// Free dense weights
	for (l = 0; l < 3; l++) {
		free(wd[l]);
		free(bd[l]);
	}

	// Free memblocks
	free(mem_block1);
	free(mem_block2);

	clReleaseMemObject(dev_zeropad);

	clReleaseMemObject(dev_matrix);
	clReleaseMemObject(dev_weights);
	clReleaseMemObject(dev_out);

}


void read_weights(char *in_file, int lvls) {
	float dval;
	int i, j, k, l, z;
	FILE *iin;
	int total_lvls_read = 0;

	iin = fopen(in_file, "r");
	if (iin == NULL) {
		printf("File %s absent\n", in_file);
		exit(1);
	}
	
	// Reading convolution weights (store them flipped from begining)
	for (z = 0; z < 13; z++) {
		if (total_lvls_read >= lvls && lvls != -1)
			break;
		printf("Read conv block %d weights\n", z);
		for (i = 0; i < cshape[z][0]; i++) {
			for (j = 0; j < cshape[z][1]; j++) {
				for (k = 0; k < cshape[z][2]; k++) {
					for (l = 0; l < cshape[z][3]; l++) {
						fscanf(iin, "%f", &dval);
						wc[z][i * cshape[z][1] * cshape[z][2] * cshape[z][3] +
							  j * cshape[z][2] * cshape[z][3] +
							  (CONV_SIZE - k  - 1)* cshape[z][3] +
							  CONV_SIZE - l - 1] = dval;
					}
				}
			}
		}
		for (i = 0; i < cshape[z][0]; i++) {
			fscanf(iin, "%f", &dval);
			bc[z][i] = dval;
		}
		total_lvls_read += 1;
	}

	// Reading dense weights
	for (z = 0; z < 3; z++) {
		if (total_lvls_read >= lvls && lvls != -1)
			break;
		printf("Read dense block %d weights\n", z);
		for (i = 0; i < dshape[z][0]; i++) {
			for (j = 0; j < dshape[z][1]; j++) {
				fscanf(iin, "%f", &dval);
				wd[z][i * dshape[z][1] + j] = dval;
			}
		}
		for (i = 0; i < dshape[z][1]; i++) {
			fscanf(iin, "%f", &dval);
			bd[z][i] = dval;
		}
		total_lvls_read += 1;
	}

	fclose(iin);
}


void normalize_image() {
	int i, j, l;
	float coef[3] = { 103.939, 116.779, 123.68 };

	for (l = 0; l < 3; l++) {
		for (i = 0; i < SIZE; i++) {
			for (j = 0; j < SIZE; j++) {
				image[IMAGE_INDEX(l, i, j)] -= coef[l];
			}
		}
	}
}

void store_image(const unsigned char *input)
{
	int i, j, l;

	for (l = 0; l < 3; l++) {
		for (i = 0; i < SIZE; i++) {
			for (j = 0; j < SIZE; j++) {
				image[IMAGE_INDEX(l, i, j)] = input[j * SIZE * 3 + i * 3 + 2 - l]; 
			}
		}
	}
}


void convolution_3_x_3(int size, float matrix[][size], float kernel[][CONV_SIZE],
                       float out[][size]) {
	int i, j;
	float sum;
	float zeropad[SIZE + 2][SIZE + 2] = { {0.} };

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			zeropad[i + 1][j + 1] = matrix[i][j];
		}
	}


	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			sum = zeropad[i][j] * kernel[0][0] +
				zeropad[i + 1][j] * kernel[1][0] +
				zeropad[i + 2][j] * kernel[2][0] +
				zeropad[i][j + 1] * kernel[0][1] +
				zeropad[i + 1][j + 1] * kernel[1][1] +
				zeropad[i + 2][j + 1] * kernel[2][1] +
				zeropad[i][j + 2] * kernel[0][2] +
				zeropad[i + 1][j + 2] * kernel[1][2] +
				zeropad[i + 2][j + 2] * kernel[2][2];
			out[i][j] += sum;
		}
	}
	
}


void convolution_3_x_3_pll(int size, int output_it, int output_offset, int input_depth) {	

	cl_int error;

	//zeropad

	//Leeg maken van zeropad
	float pattrn[] = {0};
	ocl_err(clEnqueueFillBuffer(g_command_queue, dev_zeropad,
									pattrn, 1, 0,sizeof(cl_float) * (SIZE + 2)*(SIZE + 2) * cshape[12][0],
									0,NULL,NULL));

	int arg_num = 0;
	ocl_err(clSetKernelArg(kernelZeroPad, arg_num++, sizeof(cl_mem), &dev_matrix));
	ocl_err(clSetKernelArg(kernelZeroPad, arg_num++, sizeof(cl_int), &size));
	ocl_err(clSetKernelArg(kernelZeroPad, arg_num++, sizeof(cl_mem), &dev_zeropad));



	// Call kernel
	size_t global_work_sizes[] = {size, size, input_depth};
	ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernelZeroPad, 3, NULL,
				global_work_sizes, NULL, 0, NULL, NULL));


	//inner 3x3

	// Set kernel arguments
	arg_num = 0;
	ocl_err(clSetKernelArg(kernel_inner_3x3, arg_num++, sizeof(cl_mem), &dev_zeropad));
	// ocl_err(clSetKernelArg(kernel, arg_num++, sizeof(cl_mem), &dev_matrix));
	ocl_err(clSetKernelArg(kernel_inner_3x3, arg_num++, sizeof(cl_int), &size));
	ocl_err(clSetKernelArg(kernel_inner_3x3, arg_num++, sizeof(cl_mem), &dev_weights));
	ocl_err(clSetKernelArg(kernel_inner_3x3, arg_num++, sizeof(cl_mem), &dev_out));
	ocl_err(clSetKernelArg(kernel_inner_3x3, arg_num++, sizeof(cl_int), &output_offset));
	ocl_err(clSetKernelArg(kernel_inner_3x3, arg_num++, sizeof(cl_int), &input_depth));
	ocl_err(clSetKernelArg(kernel_inner_3x3, arg_num++, sizeof(cl_int), &output_it));


	// Call kernel
	ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernel_inner_3x3, 3, NULL,
				global_work_sizes, NULL, 0, NULL, NULL));
	
}

void add_bias_and_relu(int size, float out[][size], float bs) {
	int i, j;

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			out[i][j] += bs;
			if (out[i][j] < 0)
				out[i][j] = 0.0;
		}
	}
}

void add_bias_and_relu_pll(int size, float out[][size], float bs, int output_offset) {
	cl_int error;

	int arg_num = 0;
	ocl_err(clSetKernelArg(kernelAddbs, arg_num++, sizeof(cl_int), &size));
	ocl_err(clSetKernelArg(kernelAddbs, arg_num++, sizeof(cl_mem), &dev_out));
	ocl_err(clSetKernelArg(kernelAddbs, arg_num++, sizeof(cl_float), &bs));
	ocl_err(clSetKernelArg(kernelAddbs, arg_num++, sizeof(cl_int), &output_offset));

	// Call kernel
	size_t global_work_sizes[] = {size, size};
	ocl_err(clEnqueueNDRangeKernel(g_command_queue, kernelAddbs, 2, NULL,
				global_work_sizes, NULL, 0, NULL, NULL));

}

//laatste lagen traag doordat diepte heel hoog
//opdit moment wacht
void convolution_layer(int feature_size, int input_depth, int output_depth,
					   float *input_features, float *layer_weights, float *layer_biases, float *output_features)
{
	cl_int error;

	error = clEnqueueWriteBuffer(g_command_queue, dev_weights, CL_TRUE, 0,
				cshape[level][0] * cshape[level][1] * cshape[level][2] * cshape[level][3] * sizeof(cl_float), layer_weights, 0, NULL, NULL);
	ocl_err(error);

	error = clEnqueueWriteBuffer(g_command_queue, dev_matrix, CL_TRUE, 0,
				sizeof(cl_float) * feature_size * feature_size * input_depth, input_features, 0, NULL, NULL);
	ocl_err(error);
	error = clEnqueueWriteBuffer(g_command_queue, dev_out, CL_TRUE, 0,
				output_depth * feature_size * feature_size * sizeof(cl_float), output_features, 0, NULL, NULL);
	ocl_err(error);
	for (int output_it = 0; output_it < output_depth; output_it++) {
			
		convolution_3_x_3_pll(feature_size, output_it, output_it * feature_size * feature_size, input_depth);

		add_bias_and_relu_pll(feature_size,
							&output_features[output_it * feature_size * feature_size],
							layer_biases[output_it], output_it * feature_size * feature_size);
	}
	ocl_err(clFinish(g_command_queue));
	ocl_err(clEnqueueReadBuffer(g_command_queue, dev_out, CL_TRUE,
				0, output_depth * feature_size * feature_size * sizeof(cl_float), output_features, 0, NULL, NULL));
}

void add_bias_and_relu_flatten(float *out, float *bs, int size, int relu) {
	int i;
	for (i = 0; i < size; i++) {
		out[i] += bs[i];
		if (relu == 1) {
			if (out[i] < 0)
				out[i] = 0.0;
		}
	}
}


float max_of_4(float a, float b, float c, float d) {
	if (a >= b && a >= c && a >= d) {
		return a;
	}
	if (b >= c && b >= d) {
		return b;
	}
	if (c >= d) {
		return c;
	}
	return d;
}


void maxpooling(int size, int depth, float *feature_map) {
	int i, j, d;

	//dit
	float *temp_output = malloc(depth * (size / 2) * (size / 2) * sizeof(float));
	for (d = 0; d < depth; d++) {
		float *in = &feature_map[d * size * size];
		float *out = &temp_output[d * (size / 2) * (size / 2)];
		for (i = 0; i < size; i+=2) {
			for (j = 0; j < size; j+=2) {
				out[(i / 2) * (size / 2) + j / 2] = max_of_4(in[i * size + j],
													   in[(i + 1) * size + j],
													   in[i * size + j + 1],
													   in[(i + 1) * size + j + 1]);
			}
		}
	}
	memcpy(feature_map, temp_output, depth * (size / 2) * (size / 2) * sizeof(float));
	//dit
	free(temp_output);
}


void flatten(int sh0, int sh1, int sh2, float in[sh0][sh1][sh2], float *out) {
	int i, j, k, total = 0;
	for (i = 0; i < sh0; i++) {
		for (j = 0; j < sh1; j++) {
			for (k = 0; k < sh2; k++) {
				out[total] = in[i][j][k];
				total += 1;
			}
		}
	}
}


void dense(float *in, float *weights, float *out, int sh_in, int sh_out) {
	int i, j;
	for (i = 0; i < sh_out; i++) {
		float sum = 0.0;
		for (j = 0; j < sh_in; j++) {
			sum += in[j] * weights[j * sh_out + i];
		}
		out[i] = sum;
	}
}


void softmax(float *out, int sh_out) {
	int i;
	float max_val, sum;
	max_val = out[0];
	for (i = 1; i < sh_out; i++) {
		if (out[i] > max_val)
			max_val = out[i];
	}
	sum = 0.0;
	for (i = 0; i < sh_out; i++) {
		out[i] = exp(out[i] - max_val);
		sum += out[i];
	}
	for (i = 0; i < sh_out; i++) {
		out[i] /= sum;
	}
}

void get_VGG16_predict(int only_convolution) {
	int cur_size;

	// Init intermediate memory
	reset_mem_block(mem_block1);
	reset_mem_block(mem_block2);

	// Layer 1 (Convolution 3 -> 64)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L1",1);
	#else
		time_measure_start("L1");
	#endif
	
	level = 0;
	cur_size = SIZE;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  image, wc[level], bc[level], mem_block1);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L1",1);
	#else
		time_measure_stop_and_print("L1");
	#endif
	

	//FILE* f = fopen("debug1.dat","w");
	//fwrite(mem_block1,MEM_BLOCK_DEPTH * SIZE * SIZE * sizeof(float),1,f);
	//fclose(f);
	
	// Layer 2 (Convolution 64 -> 64)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L2",2);
	#else
		time_measure_start("L2");
	#endif
	
	level = 1;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2);
	reset_mem_block(mem_block1);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L2",2);
	#else
		time_measure_stop_and_print("L2");
	#endif
	
	
	// Layer 3 (MaxPooling)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L3",3);
	#else
		time_measure_start("L3");
	#endif
	
	maxpooling(cur_size, cshape[level][0], mem_block2);
	cur_size /= 2;
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L3",3);
	#else
		time_measure_stop_and_print("L3");
	#endif
	
	
	// Layer 4 (Convolution 64 -> 128)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L4",4);
	#else
		time_measure_start("L4");
	#endif
	
	level = 2;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1);
	reset_mem_block(mem_block2);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L4",4);
	#else
		time_measure_stop_and_print("L4");
	#endif
	

	// Layer 5 (Convolution 128 -> 128)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L5",5);
	#else
		time_measure_start("L5");
	#endif
	
	level = 3;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2);
	reset_mem_block(mem_block1);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L5",5);
	#else
		time_measure_stop_and_print("L5");
	#endif
	
	
	// Layer 6 (MaxPooling)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L6",6);
	#else
		time_measure_start("L6");
	#endif
	
	maxpooling(cur_size, cshape[level][0], mem_block2);
	cur_size /= 2;
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L6",6);
	#else
		time_measure_stop_and_print("L6");
	#endif
	

	// Layer 7 (Convolution 128 -> 256)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L7",7);
	#else
		time_measure_start("L7");
	#endif
	
	level = 4;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1);
	reset_mem_block(mem_block2);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L7",7);
	#else
		time_measure_stop_and_print("L7");
	#endif
	

	// Layer 8 (Convolution 256 -> 256)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L8",8);
	#else
		time_measure_start("L8");
	#endif
	
	level = 5;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2);
	reset_mem_block(mem_block1);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L8",8);
	#else
		time_measure_stop_and_print("L8");
	#endif
	

	// Layer 9 (Convolution 256 -> 256)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L9",9);
	#else
		time_measure_start("L9");
	#endif
	
	level = 6;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1);
	reset_mem_block(mem_block2);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L9",9);
	#else
		time_measure_stop_and_print("L9");
	#endif
	
	
	// Layer 10 (MaxPooling)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L10",10);
	#else
		time_measure_start("L10");
	#endif
	
	maxpooling(cur_size, cshape[level][0], mem_block1);
	cur_size /= 2;
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L10",10);
	#else
		time_measure_stop_and_print("L10");
	#endif
	
	
	// Layer 11 (Convolution 256 -> 512)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L11",11);
	#else
		time_measure_start("L11");
	#endif
	
	level = 7;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2);
	reset_mem_block(mem_block1);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L11",11);
	#else
		time_measure_stop_and_print("L11");
	#endif
	

	// Layer 12 (Convolution 512 -> 512)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L12",12);
	#else
		time_measure_start("L12");
	#endif
	
	level = 8;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1);
	reset_mem_block(mem_block2);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L12",12);
	#else
		time_measure_stop_and_print("L12");
	#endif
	

	// Layer 13 (Convolution 512 -> 512)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L13",13);
	#else
		time_measure_start("L13");
	#endif
	
	level = 9;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2);
	reset_mem_block(mem_block1);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L13",13);
	#else
		time_measure_stop_and_print("L13");
	#endif
	
	
	// Layer 14 (MaxPooling)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L14",14);
	#else
		time_measure_start("L14");
	#endif
	
	maxpooling(cur_size, cshape[level][0], mem_block2);
	// for (i = 0; i < cshape[level][0]; i++) {
	// 	maxpooling(cur_size, &mem_block2[i * cur_size * cur_size]);
	// }
	cur_size /= 2;
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L14",14);
	#else
		time_measure_stop_and_print("L14");
	#endif
	
	
	// Layer 15 (Convolution 512 -> 512)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L15",15);
	#else
		time_measure_start("L15");
	#endif
	
	level = 10;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1);
	reset_mem_block(mem_block2);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L15",15);
	#else
		time_measure_stop_and_print("L15");
	#endif
	

	// Layer 16 (Convolution 512 -> 512)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L16",16);
	#else
		time_measure_start("L16");
	#endif
	
	level = 11;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block1, wc[level], bc[level], mem_block2);
	reset_mem_block(mem_block1);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L16",16);
	#else
		time_measure_stop_and_print("L16");
	#endif
	

	// Layer 17 (Convolution 512 -> 512)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L17",17);
	#else
		time_measure_start("L17");
	#endif
	
	level = 12;
	convolution_layer(cur_size, cshape[level][1], cshape[level][0],
					  mem_block2, wc[level], bc[level], mem_block1);
	reset_mem_block(mem_block2);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L17",17);
	#else
		time_measure_stop_and_print("L17");
	#endif
	
	
	// Layer 18 (MaxPooling)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L18",18);
	#else
		time_measure_start("L18");
	#endif
	
	maxpooling(cur_size, cshape[level][0], mem_block1);
	cur_size /= 2;
	
	mem_block1_dense = mem_block2;
	mem_block2_dense = mem_block1;
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L18",18);
	#else
		time_measure_stop_and_print("L18");
	#endif
	

	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L19",19);
	#else
		time_measure_start("L19");
	#endif
	
	reset_mem_block_dense(mem_block1_dense);
	// Layer 19 (Flatten)
	flatten(cshape[level][0], cur_size, cur_size, mem_block1, mem_block1_dense);
	if (only_convolution == 1) {
		return;
	}
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L19",19);
	#else
		time_measure_stop_and_print("L19");
	#endif
	


	//reset_mem_block_dense(mem_block2_dense);

	// Layer 20 (Dense)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L20",20);
	#else
		time_measure_start("L20");
	#endif
	
	level = 0;
	dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);
	reset_mem_block_dense(mem_block1_dense);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L20",20);
	#else
		time_measure_stop_and_print("L20");
	#endif
	

	// Layer 21 (Dense)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L21",21);
	#else
		time_measure_start("L21");
	#endif
	
	level = 1;
	dense(mem_block2_dense, wd[level], mem_block1_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block1_dense, bd[level], dshape[level][1], 1);
	reset_mem_block_dense(mem_block2_dense);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L21",21);
	#else
		time_measure_stop_and_print("L21");
	#endif
	

	
	// Layer 22 (Dense)
	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("L22",22);
	#else
		time_measure_start("L22");
	#endif
	
	level = 2;
	dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);
	softmax(mem_block2_dense, dshape[level][1]);
	// dump_memory_structure_dense_to_file(mem_block2_dense, dshape[level][1]);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("L22",22);
	#else
		time_measure_stop_and_print("L22");
	#endif
	
	
	return;
}


void output_predictions() {
	float max = 0.f;
	int max_idx = 0;
	for (int i = 0; i < dshape[2][1]; i++) {
		 if (max <  mem_block2_dense[i]) {
			 max = mem_block2_dense[i];
			 max_idx = i;
		 }
	}

	printf("Prediction: %s (score = %f)\n", imagenet_labels[max_idx], max);
}



int main(int argc, char *argv[]) {
	char buf[1024];
	char *weights_file;
	char *output_file;
	int lvls = -1;
	int only_convolution = 0;

	if (argc != 3) {
		printf("Usage: <program> <weights file> <image>\n");
		return 0;
	}
	weights_file = argv[1];
	if (argc == 5) {
		lvls = 13;
		only_convolution = 1;
	}
	cl_platform_id platform = ocl_select_platform();
    cl_device_id device = ocl_select_device(platform);
    init_ocl(device);
	create_program("kernel.cl", "");


	init_memory();
	read_weights(weights_file, lvls);

	int width, height, num_channels;
    unsigned char *input_image = stbi_load(argv[2], &width, &height, &num_channels, 3);
	store_image(input_image);
	normalize_image();

	#ifdef SIMPEL_TIME_UTILS
		time_measure_start("prediction",0);
	#else
		time_measure_start("prediction");
	#endif
	
	get_VGG16_predict(only_convolution);
	#ifdef SIMPEL_TIME_UTILS
		time_measure_stop_and_print("prediction",0);
	#else
		time_measure_stop_and_print("prediction");
	#endif
	

	output_predictions();


	free_memory();
	return 0;
}

