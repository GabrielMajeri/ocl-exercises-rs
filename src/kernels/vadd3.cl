/// This kernel computes `dest = a + b + c`, for `a`, `b` and `c` vectors of floats.
__kernel void vadd3(
    const __global float* a,
    const __global float* b,
    const __global float* c,
    __global float* dest) {
    // Get the ID of the current work item
    int id = get_global_id(0);

    // Perform the addition of the work item this execution is responsible for
    dest[id] = a[id] + b[id] + c[id];
}
