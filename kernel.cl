//Tijs Van Kampen en Lucas Van Laer
//versie 4d
#define SIZE 224
#define NUM_CHANNELS 3
#define CONV_SIZE 3
#define kern_SIZE 3
#define NUM_LAYERS 13
#define NUM_DENSE 3

#define IMAGE_INDEX(l, x, y) ((l) * (SIZE) * (SIZE) + (y) * (SIZE) + (x))
#define PLANE_INDEX(l) ((l) * (SIZE) * (SIZE))
#define MEM_BLOCK_DEPTH 512

inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal,
                newVal.intVal) != prevVal.intVal);
}

__kernel void inner_convolutie3x3(__global float *zeropad, int size,
                                  __global float *kernFull, __global float *outFull, int input_depth, int output_depth) {

	float sum = 0;

	const int j = get_global_id(0);
	const int i = get_global_id(1);
        const int u = get_global_id(2)%input_depth;//input_it
        const int o = get_global_id(2)/input_depth;//output_it



        int weights_offset = o * input_depth * CONV_SIZE * CONV_SIZE + u * CONV_SIZE * CONV_SIZE;
        
	sum = zeropad[u*(SIZE+2)*(SIZE + 2) + i * (SIZE + 2) + j] * kernFull[0 + weights_offset] +
		zeropad[u*(SIZE+2)*(SIZE + 2) + (i + 1) * (SIZE + 2) + j] * kernFull[1 * CONV_SIZE + weights_offset] +
		zeropad[u*(SIZE+2)*(SIZE + 2) + (i + 2) * (SIZE + 2) + j] * kernFull[2 * CONV_SIZE + weights_offset] +
		zeropad[u*(SIZE+2)*(SIZE + 2) + i * (SIZE + 2) + j + 1] * kernFull[1 + weights_offset] +
		zeropad[u*(SIZE+2)*(SIZE + 2) + (i + 1) * (SIZE + 2) + j + 1] * kernFull[1 + CONV_SIZE + weights_offset] +
		zeropad[u*(SIZE+2)*(SIZE + 2) + (i + 2) * (SIZE + 2) + j + 1] * kernFull[1 + 2 * CONV_SIZE + weights_offset] +
		zeropad[u*(SIZE+2)*(SIZE + 2) + i * (SIZE + 2) + j + 2] * kernFull[2 + weights_offset] +
		zeropad[u*(SIZE+2)*(SIZE + 2) + (i + 1) * (SIZE + 2) + j + 2] * kernFull[2 + 1 * CONV_SIZE + weights_offset] +
		zeropad[u*(SIZE+2)*(SIZE + 2) + (i + 2) * (SIZE + 2) + j + 2] * kernFull[2 + 2 * CONV_SIZE + weights_offset];

        AtomicAdd(&outFull[size * i + j + o*size*size], sum);
}

__kernel void zeropadConv(__global float *matrixFull, int size,
                          __global float *zeropad) {
	const int j = get_global_id(0);
	const int i = get_global_id(1);
        const int u = get_global_id(2);

	zeropad[u*(SIZE + 2)*(SIZE + 2) + (i + 1) * (SIZE + 2) + j + 1] = matrixFull[i * size + j + u * size * size];
}


__kernel void add_bias_and_relu(int size, __global float *outFull, float bs, int output_offset) {

	const int j = get_global_id(0);
	const int i = get_global_id(1);

	outFull[i * size + j + output_offset] += bs;
	if (outFull[i * size + j + output_offset] < 0.0f)
		outFull[i * size + j + output_offset] = 0.0f;
}