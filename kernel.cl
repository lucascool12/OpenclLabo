#define SIZE 224
#define NUM_CHANNELS 3
#define CONV_SIZE 3
#define kern_SIZE 3
#define NUM_LAYERS 13
#define NUM_DENSE 3

#define IMAGE_INDEX(l, x, y) ((l) * (SIZE) * (SIZE) + (y) * (SIZE) + (x))
#define PLANE_INDEX(l) ((l) * (SIZE) * (SIZE))
#define MEM_BLOCK_DEPTH 512

__kernel void inner_convolutie3x3(__global float *zeropad, int size,
                                  __global float *kernFull, __global float *outFull,
                                  int weights_offset, int output_offset) {

	float sum = 0;

	const int j = get_global_id(0);
	const int i = get_global_id(1);

	sum = zeropad[i * (SIZE + 2) + j] * kernFull[0 + weights_offset] +
		zeropad[(i + 1) * (SIZE + 2) + j] * kernFull[1 * CONV_SIZE + weights_offset] +
		zeropad[(i + 2) * (SIZE + 2) + j] * kernFull[2 * CONV_SIZE + weights_offset] +
		zeropad[i * (SIZE + 2) + j + 1] * kernFull[1 + weights_offset] +
		zeropad[(i + 1) * (SIZE + 2) + j + 1] * kernFull[1 + CONV_SIZE + weights_offset] +
		zeropad[(i + 2) * (SIZE + 2) + j + 1] * kernFull[1 + 2 * CONV_SIZE + weights_offset] +
		zeropad[i * (SIZE + 2) + j + 2] * kernFull[2 + weights_offset] +
		zeropad[(i + 1) * (SIZE + 2) + j + 2] * kernFull[2 + 1 * CONV_SIZE + weights_offset] +
		zeropad[(i + 2) * (SIZE + 2) + j + 2] * kernFull[2 + 2 * CONV_SIZE + weights_offset];
	outFull[size * i + j + output_offset] += sum;
}

__kernel void zeropadConv(__global float *matrixFull, int size,
                          __global float *zeropad , int input_offset) {
	const int j = get_global_id(0);
	const int i = get_global_id(1);

	zeropad[(i + 1) * (SIZE + 2) + j + 1] = matrixFull[i * size + j + input_offset];
}

__kernel void add_bias_and_relu(int size, __global float *outFull, float bs, int output_offset) {

	const int j = get_global_id(0);
	const int i = get_global_id(1);

	outFull[i * size + j + output_offset] += bs;
	if (outFull[i * size + j + output_offset] < 0.0f)
		outFull[i * size + j + output_offset] = 0.0f;
}