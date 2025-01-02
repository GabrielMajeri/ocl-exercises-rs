/// This kernel computes `C = A @ B`, for `A` and `B` matrices of floating-point numbers.
__kernel void matmul_opt_row(
    const int N,
    const __global float* a,
    const __global float* b,
    __global float* dest) {
    // Get the row of the destination matrix,
    // assigned to this work group.
    int row = get_global_id(0);

    const int MAX_N = 512;
    float row_copy[MAX_N];
    for (int column = 0; column < N; ++column) {
        row_copy[column] = a[row * N + column];
    }

    for (int column = 0; column < N; ++column) {
        // Compute the result of the dot product `A[row, :] @ B[:, column]`.
        float accumulator = 0.0;
        for (int k = 0; k < N; ++k) {
            accumulator += row_copy[k] * b[k * N + column];
        }
        // Store the result.
        dest[row * N + column] = accumulator;
    }
}
