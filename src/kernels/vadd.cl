__kernel void vadd(const __global float* a, const __global float* b, __global float* dest) {
    int id = get_global_id(0);

    dest[id] = a[id] + b[id];
}
