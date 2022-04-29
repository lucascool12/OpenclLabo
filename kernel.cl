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
                                  __global float *kernFull,
                                  __global float *outFull, int weights_offset,
                                  int output_offset) {

  float sum = 0;

  const int j = get_global_id(0);
  const int i = get_global_id(1);

  // const int b = get_work_dim();

  //     __global float *kern = &kernFull[CONV_SIZE * weights_offset];
  //     __global float *out = &outFull[size * output_offset];

  sum = zeropad[i * (SIZE + 2) + j] * kernFull[0 + weights_offset] +
        zeropad[(i + 1) * (SIZE + 2) + j] *
            kernFull[1 * CONV_SIZE + weights_offset] +
        zeropad[(i + 2) * (SIZE + 2) + j] *
            kernFull[2 * CONV_SIZE + weights_offset] +
        zeropad[i * (SIZE + 2) + j + 1] * kernFull[1 + weights_offset] +
        zeropad[(i + 1) * (SIZE + 2) + j + 1] *
            kernFull[1 + CONV_SIZE + weights_offset] +
        zeropad[(i + 2) * (SIZE + 2) + j + 1] *
            kernFull[1 + 2 * CONV_SIZE + weights_offset] +
        zeropad[i * (SIZE + 2) + j + 2] * kernFull[2 + weights_offset] +
        zeropad[(i + 1) * (SIZE + 2) + j + 2] *
            kernFull[2 + 1 * CONV_SIZE + weights_offset] +
        zeropad[(i + 2) * (SIZE + 2) + j + 2] *
            kernFull[2 + 2 * CONV_SIZE + weights_offset];
  outFull[size * i + j + output_offset] += sum;

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

void zeropadConvfunc(__global float *matrixFull, int size,
                     __global float *zeropad, int input_offset) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);

  // __global float *matrix = &matrixFull[size  * input_offset];

  zeropad[(i + 1) * (SIZE + 2) + j + 1] =
      matrixFull[i * size + j + input_offset];

  // zeropad[(i + 1) * (SIZE + 2) + j + 1] = matrix[i * size + j];
}

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
  } while (atomic_cmpxchg((volatile __global unsigned int *)source,
                          prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void inner_convolutie3x3_test(__global float *zeropad, int size,
                                       __global float *kernFull,
                                       __global float *outFull,
                                       int output_offset, int input_depth,
                                       int output_it) {

  float sum = 0;

  const int j = get_global_id(0);
  const int i = get_global_id(1);
  const int u = get_global_id(2);

  // const int b = get_work_dim();

  //     __global float *kern = &kernFull[CONV_SIZE * weights_offset];
  //     __global float *out = &outFull[size * output_offset];

  int weights_offset = output_it * input_depth * CONV_SIZE * CONV_SIZE +
                       u * CONV_SIZE * CONV_SIZE;

  sum = zeropad[u * (SIZE + 2) * (SIZE + 2) + i * (SIZE + 2) + j] *
            kernFull[0 + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 1) * (SIZE + 2) + j] *
            kernFull[1 * CONV_SIZE + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 2) * (SIZE + 2) + j] *
            kernFull[2 * CONV_SIZE + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + i * (SIZE + 2) + j + 1] *
            kernFull[1 + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 1) * (SIZE + 2) + j + 1] *
            kernFull[1 + CONV_SIZE + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 2) * (SIZE + 2) + j + 1] *
            kernFull[1 + 2 * CONV_SIZE + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + i * (SIZE + 2) + j + 2] *
            kernFull[2 + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 1) * (SIZE + 2) + j + 2] *
            kernFull[2 + 1 * CONV_SIZE + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 2) * (SIZE + 2) + j + 2] *
            kernFull[2 + 2 * CONV_SIZE + weights_offset];

  AtomicAdd(&outFull[size * i + j + output_offset], sum);
  // outFull[size * i + j + output_offset] += sum;

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

__kernel void inner_convolutie3x3_tester(__global float *zeropad, int size,
                                         __global float *kernFull,
                                         __global float *outFull,
                                         int input_depth, int output_depth) {

    float sum = 0;

  const int j = get_global_id(0);
  const int i = get_global_id(1);
  const int u = get_global_id(2) % input_depth; // input_it
  const int o = get_global_id(2) / input_depth; // output_it

  int weights_offset =
      o * input_depth * CONV_SIZE * CONV_SIZE + u * CONV_SIZE * CONV_SIZE;

  sum = zeropad[u * (SIZE + 2) * (SIZE + 2) + i * (SIZE + 2) + j] *
            kernFull[0 + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 1) * (SIZE + 2) + j] *
            kernFull[1 * CONV_SIZE + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 2) * (SIZE + 2) + j] *
            kernFull[2 * CONV_SIZE + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + i * (SIZE + 2) + j + 1] *
            kernFull[1 + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 1) * (SIZE + 2) + j + 1] *
            kernFull[1 + CONV_SIZE + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 2) * (SIZE + 2) + j + 1] *
            kernFull[1 + 2 * CONV_SIZE + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + i * (SIZE + 2) + j + 2] *
            kernFull[2 + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 1) * (SIZE + 2) + j + 2] *
            kernFull[2 + 1 * CONV_SIZE + weights_offset] +
        zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 2) * (SIZE + 2) + j + 2] *
            kernFull[2 + 2 * CONV_SIZE + weights_offset];

  AtomicAdd(&outFull[size * i + j + o * size * size], sum);
}

__kernel void zeropadConv_test(__global float *matrixFull, int size,
                               __global float *zeropad) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);
  const int u = get_global_id(2);

  // __global float *matrix = &matrixFull[size  * input_offset];

  zeropad[u * (SIZE + 2) * (SIZE + 2) + (i + 1) * (SIZE + 2) + j + 1] =
      matrixFull[i * size + j + u * size * size];

  // zeropad[(i + 1) * (SIZE + 2) + j + 1] = matrix[i * size + j];
}

__kernel void zeropadConv(__global float *matrixFull, int size,
                          __global float *zeropad, int input_offset) {
  const int j = get_global_id(0);
  const int i = get_global_id(1);

  // __global float *matrix = &matrixFull[size  * input_offset];

  zeropad[(i + 1) * (SIZE + 2) + j + 1] =
      matrixFull[i * size + j + input_offset];

  // zeropad[(i + 1) * (SIZE + 2) + j + 1] = matrix[i * size + j];
}

__kernel void add_bias_and_relu(int size, __global float *outFull, float bs,
                                int output_offset) {

  const int j = get_global_id(0);
  const int i = get_global_id(1);

  // __global float *out = &outFull[size * output_offset];

  outFull[i * size + j + output_offset] += bs;
  if (outFull[i * size + j + output_offset] < 0.0f)
    outFull[i * size + j + output_offset] = 0.0f;
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

__kernel void dense(__global float *in, __global float *weights,
                    __global float *out, int sh_out) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  float sum = 0.0f;

  sum = in[j] * weights[j * sh_out + i];

  AtomicAdd(&out[i], sum);
}