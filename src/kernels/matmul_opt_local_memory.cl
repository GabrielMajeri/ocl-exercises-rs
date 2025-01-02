/// This kernel computes `C = A @ B`, for `A` and `B` matrices of floating-point numbers.
__kernel void matmul_opt_local_memory(
    const int N,
    const __global float* a,
    const __global float* b,
    __global float* dest,
    __local float* column_copy) {
    // Get the row of the destination matrix,
    // assigned to this work group.
    int row = get_global_id(0);

    // Get ID and size assigned to this work group.
    // (each work group gets assigned a column of B)
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    const int MAX_N = 1024;
    float row_copy[MAX_N];
    for (int column = 0; column < N; ++column) {
        row_copy[column] = a[row * N + column];
    }

    for (int column = 0; column < N; ++column) {
        // Make a copy of the column assigned to our work group.
        for (int k = local_id; k < N; k += local_size) {
            column_copy[k] = b[k * N + column];
        }

        // Wait for all work items to finish copying the column.
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute the result of the dot product `A[row, :] @ B[:, column]`.
        float accumulator = 0.0;
        for (int k = 0; k < N; ++k) {
            accumulator += row_copy[k] * column_copy[k];
        }

        // Store the result.
        dest[row * N + column] = accumulator;

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
