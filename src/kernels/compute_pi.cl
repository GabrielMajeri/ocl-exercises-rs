/// This kernel estimates pi by integrating `4 / (1 + x^2)` over [0, 1].
///
/// Each work item integrates a strided subset of the domain, then the work
/// group reduces its partial sums in local memory and writes a single value.
__kernel void compute_pi(
    const double step_size,
    const int num_points,
    __local double* local_sums,
    __global double* group_sums
) {
    // Global ID and global size determine which points this work item will process,
    // and how many total work items there are.
    const int global_id = get_global_id(0);
    const int global_size = get_global_size(0);

    double partial_sum = 0.0;
    for (int point = global_id; point < num_points; point += global_size) {
        const double x = ((double)point + 0.5) * step_size;
        partial_sum += 4.0 / (1.0 + x * x);
    }

    // Local ID and local size are used for the reduction within the work group.
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);

    // Store the partial sum in local memory and synchronize the work group.
    local_sums[local_id] = partial_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform a reduction within the work group to sum the partial sums.
    // This is a parallel reduction pattern where we
    // iteratively halve the number of active work items,
    // giving us a logarithmic number of steps.
    for (int offset = local_size / 2; offset > 0; offset /= 2) {
        if (local_id < offset) {
            local_sums[local_id] += local_sums[local_id + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Group ID is used to index into the group_sums array for the final result of this work group.
    const int group_id = get_group_id(0);

    if (local_id == 0) {
        group_sums[group_id] = local_sums[0];
    }
}
