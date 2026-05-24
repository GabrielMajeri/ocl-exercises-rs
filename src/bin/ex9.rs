use std::f64::consts::PI;
use std::time::{Duration, Instant};

use ocl::{core::ProfilingInfo, Buffer, CommandQueueProperties, Event, ProQue};

const NUM_POINTS: usize = 128 * 1024 * 1024;
const STEP_SIZE: f64 = 1.0 / (NUM_POINTS as f64);
const WORK_GROUP_SIZE: usize = 256;
const NUM_WORK_GROUPS: usize = 256;
const GLOBAL_WORK_SIZE: usize = WORK_GROUP_SIZE * NUM_WORK_GROUPS;

fn main() {
    println!("# Exercise 9 - Estimate the value of pi numerically");
    println!("Construct a kernel which determines an approximation of pi,");
    println!("by computing the (Riemann) integral of 4/(1 + x^2).");
    compute_pi();
}

fn compute_pi() {
    println!("Computing pi on the CPU...");

    let start = Instant::now();

    let mut sum = 0.0f64;

    for i in 0..NUM_POINTS {
        let x = ((i as f64) + 0.5) * STEP_SIZE;
        sum += 4.0 / (1.0 + x * x);
    }

    let host_pi_estimate = STEP_SIZE * sum;
    let host_relative_error = (host_pi_estimate - PI).abs() / PI;

    let cpu_time = start.elapsed();

    println!("Computing pi on the OpenCL device...");

    // Load the OCL kernel source code.
    let kernel = include_str!("../kernels/compute_pi.cl");

    // Compile the kernel into a runnable program.
    let pro_que = ProQue::builder()
        .src(kernel)
        .dims(GLOBAL_WORK_SIZE)
        .queue_properties(CommandQueueProperties::PROFILING_ENABLE)
        .build()
        .expect("Failed to create OpenCL context");

    let group_sums_buffer = Buffer::<f64>::builder()
        .queue(pro_que.queue().clone())
        .len(NUM_WORK_GROUPS)
        .build()
        .expect("Failed to create group sums buffer");

    // Estimate the value of pi using OpenCL kernels
    let compute_pi_kernel = pro_que
        .kernel_builder("compute_pi")
        .arg(STEP_SIZE)
        .arg(NUM_POINTS as i32)
        .arg_local::<f64>(WORK_GROUP_SIZE)
        .arg(&group_sums_buffer)
        .build()
        .expect("Failed to compile OpenCL kernel");

    let mut compute_pi_event = Event::empty();
    unsafe {
        compute_pi_kernel
            .cmd()
            .global_work_size(GLOBAL_WORK_SIZE)
            .local_work_size(WORK_GROUP_SIZE)
            .enew(&mut compute_pi_event)
            .enq()
            .expect("Failed to enqueue compute pi kernel");
    }

    pro_que.queue().finish().unwrap();

    let compute_pi_start = compute_pi_event
        .profiling_info(ProfilingInfo::Start)
        .unwrap()
        .time()
        .unwrap();
    let compute_pi_end = compute_pi_event
        .profiling_info(ProfilingInfo::End)
        .unwrap()
        .time()
        .unwrap();
    let compute_pi_duration = Duration::from_nanos(compute_pi_end - compute_pi_start);

    // Read back one partial result per work group and
    // finish the small final reduction on the host.
    let mut group_sums = vec![0.0; NUM_WORK_GROUPS];

    let start = Instant::now();

    group_sums_buffer.read(&mut group_sums).enq().unwrap();
    let total_sum: f64 = group_sums.iter().sum();

    let host_reduction_duration = start.elapsed();

    let opencl_time = compute_pi_duration + host_reduction_duration;

    println!("Computation finished, displaying results...");

    let device_pi_estimate = STEP_SIZE * total_sum;
    let device_relative_error = (device_pi_estimate - PI).abs() / PI;

    println!(
        "Estimate of pi (computed on the CPU): {} (relative error: {:E})",
        host_pi_estimate, host_relative_error
    );
    println!(
        "Estimate of pi (computed on the device): {} (relative error: {:E})",
        device_pi_estimate, device_relative_error
    );
    println!("Truer value of pi (from math library): {}", PI);

    println!("CPU time: {:?}", cpu_time);
    println!("OpenCL time: {:?}", opencl_time);
}
