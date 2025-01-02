use std::time::Instant;

use ocl::flags::MemFlags;
use ocl::{Buffer, ProQue};

use rand::distributions::Standard;
use rand::{rngs::StdRng, Rng, SeedableRng};

use float_cmp::ApproxEqRatio;

const SEED: u64 = 7;
const N: usize = 64;
const SIZE: usize = N * N;

fn main() {
    println!("# Exercises 7 - Optimized matrix multiplication");
    println!("Construct a kernel which multiplies two square matrices,");
    println!("where each work item computes a row of the result matrix");
    matmul_opt_row();
}

pub fn matmul_opt_row() {
    // Generate some matrices containing random data.
    let mut rng = StdRng::seed_from_u64(SEED);

    fn generate_data(rng: &mut StdRng) -> Vec<f32> {
        rng.sample_iter(Standard).take(SIZE).collect()
    }

    let a = generate_data(&mut rng);
    let b = generate_data(&mut rng);

    // Compute the result using the CPU.
    let mut correct_result = vec![0.0; SIZE];

    let start = Instant::now();
    for i in 0..N {
        let row_copy = a[i * N..(i + 1) * N].to_owned();
        for j in 0..N {
            let mut accumulator = 0.0;
            for k in 0..N {
                unsafe {
                    let x = *row_copy.get_unchecked(k);
                    let y = *b.get_unchecked(k * N + j);
                    accumulator += x * y;
                }
            }
            correct_result[i * N + j] = accumulator;
        }
    }
    let end = Instant::now();
    let cpu_time = end.duration_since(start);

    // Load the OCL kernel source code.
    let kernel = include_str!("../kernels/matmul_opt_row.cl");

    // Compile the kernel into a runnable program.
    let pro_que = ProQue::builder()
        .src(kernel)
        .dims(N)
        .build()
        .expect("Failed to create OpenCL context");

    // Create buffers for the input vectors.
    let build_input_buffer = |data: &[f32]| {
        Buffer::<f32>::builder()
            .queue(pro_que.queue().clone())
            .len(SIZE)
            .flags(MemFlags::new().read_only().copy_host_ptr())
            .copy_host_slice(data)
            .build()
            .expect("Failed to create buffer")
    };

    let a_buf = build_input_buffer(&a);
    let b_buf = build_input_buffer(&b);

    let build_destination_buffer = || {
        Buffer::<f32>::builder()
            .queue(pro_que.queue().clone())
            .len(SIZE)
            .flags(MemFlags::new().write_only())
            .build()
            .expect("Failed to create destination buffer")
    };

    let dest_buf = build_destination_buffer();

    // Execute `dest_buf = a_buf @ b_buf`
    let matmul = pro_que
        .kernel_builder("matmul_opt_row")
        .arg(N as i32)
        .arg(&a_buf)
        .arg(&b_buf)
        .arg(&dest_buf)
        .build()
        .expect("Failed to compile OpenCL kernel");

    let start = Instant::now();
    unsafe {
        matmul.enq().expect("Failed to execute OpenCL kernel");
    }
    let end = Instant::now();
    let opencl_time = end.duration_since(start);

    // Read back the result into `dest`.
    let mut dest = vec![0.0; SIZE];
    dest_buf.read(&mut dest).enq().unwrap();

    let (row, column) = (rng.gen_range(0..N), rng.gen_range(0..N));

    let result = dest[row * N + column];
    let correct_result = correct_result[row * N + column];

    let correct = if result.approx_eq_ratio(&correct_result, 0.001) {
        '✓'
    } else {
        '❌'
    };

    println!(
        "(A[{}, :] @ B[:, {}]) = {} {}",
        row, column, result, correct
    );

    println!("CPU time: {:?}", cpu_time);
    println!("OpenCL time: {:?}", opencl_time);
}
