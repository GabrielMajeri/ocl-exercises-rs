__kernel void vadd(const __global float* a, const __global float* b, __global float* dest) {
    // Get the ID of the current work item
    int id = get_global_id(0);

    // Perform the addition of the work item this execution is responsible for
    dest[id] = a[id] + b[id];
}
