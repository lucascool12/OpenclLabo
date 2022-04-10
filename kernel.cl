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
                                  __global float *kern, __global float *out/*,
                                  int weights_offset, int output_offset*/) {

	float sum = 0;

	const int j = get_global_id(0);
	const int i = get_global_id(1);

  //   __global float *kern = &kernFull[CONV_SIZE * CONV_SIZE * weights_offset];
  //   __global float *out = &outFull[size * size * output_offset];

  //   float zeropad[(SIZE + 2) * 3] = {{0.}};
  //   int eersteLeeg = 1;
  //   if(i == 0)
  //         eersteLeeg = 0;

  //   for (int ii = i; ii < 2 - eersteLeeg; ii++) {
  //     for (int jj = j; jj < size; jj++) {
  //             zeropad[(ii - i + 1)*(SIZE + 2) + jj - j + eersteLeeg] =
  //             matrix[ii*(size) + jj];
  //     }
  //   }

	sum = zeropad[i * (SIZE + 2) + j] * kern[0] +
		zeropad[(i + 1) * (SIZE + 2) + j] * kern[1 * CONV_SIZE] +
		zeropad[(i + 2) * (SIZE + 2) + j] * kern[2 * CONV_SIZE] +
		zeropad[i * (SIZE + 2) + j + 1] * kern[1] +
		zeropad[(i + 1) * (SIZE + 2) + j + 1] * kern[1 + CONV_SIZE] +
		zeropad[(i + 2) * (SIZE + 2) + j + 1] * kern[1 + 2 * CONV_SIZE] +
		zeropad[i * (SIZE + 2) + j + 2] * kern[2] +
		zeropad[(i + 1) * (SIZE + 2) + j + 2] * kern[2 + 1 * CONV_SIZE] +
		zeropad[(i + 2) * (SIZE + 2) + j + 2] * kern[2 + 2 * CONV_SIZE];
	out[size * i + j] += sum;

  //   if(i > 0 && j > 0)
  //         sum += zeropad[i * (SIZE + 2) + j] * kern[0];
  //   if(j > 0)
  //         sum += zeropad[(i + 1) * (SIZE + 2) + j] * kern[1 * CONV_SIZE];
  //   if(j > 0)
  //         sum += zeropad[(i + 2) * (SIZE + 2) + j] * kern[2 * CONV_SIZE];

  //   sum += zeropad[i * (SIZE + 2) + j + 1] * kern[1] +
  //         zeropad[(i + 1) * (SIZE + 2) + j + 1] * kern[1 + CONV_SIZE] +
  //         zeropad[(i + 2) * (SIZE + 2) + j + 1] * kern[1 + 2 * CONV_SIZE] +
  //         zeropad[i * (SIZE + 2) + j + 2] * kern[2] +
  //         zeropad[(i + 1) * (SIZE + 2) + j + 2] * kern[2 + 1 * CONV_SIZE] +
  //         zeropad[(i + 2) * (SIZE + 2) + j + 2] * kern[2 + 2 * CONV_SIZE];
  //   out[size * i + j] += sum;

  //         sum = zeropad[i * (SIZE + 2) + j] * kern[0] +
  //         zeropad[(i + 1) * (SIZE + 2) + j] * kern[1 * CONV_SIZE] +
  //         zeropad[(i + 2) * (SIZE + 2) + j] * kern[2 * CONV_SIZE] +
  //         zeropad[i * (SIZE + 2) + j + 1] * kern[1] +
  //         zeropad[(i + 1) * (SIZE + 2) + j + 1] * kern[1 + CONV_SIZE] +
  //         zeropad[(i + 2) * (SIZE + 2) + j + 1] * kern[1 + 2 * CONV_SIZE] +
  //         zeropad[i * (SIZE + 2) + j + 2] * kern[2] +
  //         zeropad[(i + 1) * (SIZE + 2) + j + 2] * kern[2 + 1 * CONV_SIZE] +
  //         zeropad[(i + 2) * (SIZE + 2) + j + 2] * kern[2 + 2 * CONV_SIZE];
  //   out[size * i + j] += sum;
}

__kernel void zeropadConv(__global float *matrix, int size,
                          __global float *zeropad /*, int input_offset*/) {
	const int i = get_global_id(0);
	const int j = get_global_id(1);

	// __global float *matrix = &matrixFull[size * size * input_offset];

	zeropad[(i + 1) * (SIZE + 2) + j + 1] = matrix[i * size + j];
}

__kernel void add_bias_and_relu(int size, __global float *out, float bs) {

	const int i = get_global_id(0);
	const int j = get_global_id(1);

	out[i * size + j] += bs;
	if (out[i * size + j] < 0.0f)
		out[i * size + j] = 0.0f;
}

/*kernel void convolutie3x3(
    __global int *size,
    __global float *matrix,
    __global float *kern,
    __global float *out
){
    const int inp = get_global_id(0);
    const int outp = get_global_id(1);
    int i, j;
        float sum;
        float zeropad[SIZE + 2][SIZE + 2] = { {0.} };

        for (i = 0; i < *size; i++) {
                for (j = 0; j < *size; j++) {
                        zeropad[i + 1][j + 1] = matrix[*size*i+j];
                }
        }

        for (i = 0; i < *size; i++) {
                for (j = 0; j < *size; j++) {
                        sum = zeropad[i][j] * kern[0] +
                                zeropad[i + 1][j] * kern[1] +
                                zeropad[i + 2][j] * kern[2] +
                                zeropad[i][j + 1] * kern[1] +
                                zeropad[i + 1][j + 1] * kern[1+CONV_SIZE] +
                                zeropad[i + 2][j + 1] * kern[1+2*CONV_SIZE] +
                                zeropad[i][j + 2] * kern[2] +
                                zeropad[i + 1][j + 2] * kern[2+1*CONV_SIZE] +
                                zeropad[i + 2][j + 2] * kern[2+2*CONV_SIZE];
                        out[*size*i+j] += sum;
                }
        int i, j;

            for (i = 0; i < *size; i++) {
                    for (j = 0; j < *size; j++) {
                            out[*size*i+j] += bs;
                            if (out[*size*i+j] < 0)
                                out[*size*i+j] = 0.0;
                    }
            }
        }
}*/