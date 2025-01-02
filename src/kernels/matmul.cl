/// This kernel computes `C = A @ B`, for `A` and `B` matrices of floating-point numbers.
__kernel void matmul(
    const int N,
    const __global float* a,
    const __global float* b,
    __global float* dest) {
    // Get the row and column of the destination matrix,
    // assigned to this work item.
    int row = get_global_id(0);
    int column = get_global_id(1);

    // Compute the result of the dot product `A[row, :] @ B[:, column]`.
    float accumulator = 0.0;
    for (int k = 0; k < N; ++k) {
        accumulator += a[row * N + k] * b[k * N + column];
    }

    // Store the result.
    dest[row * N + column] = accumulator;
}
