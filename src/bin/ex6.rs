use ocl::flags::MemFlags;
use ocl::{Buffer, ProQue};

use rand::distributions::Standard;
use rand::{rngs::StdRng, Rng, SeedableRng};

use float_cmp::ApproxEqRatio;

const SEED: u64 = 42;
const N: usize = 64;
const SIZE: usize = N * N;

fn main() {
    println!("# Exercises 6 - Matrix multiplication");
    println!("Construct a kernel which multiplies two square matrices");
    matmul();
}

pub fn matmul() {
    // Generate some matrices containing random data.
    let mut rng = StdRng::seed_from_u64(SEED);

    fn generate_data(rng: &mut StdRng) -> Vec<f32> {
        rng.sample_iter(Standard).take(SIZE).collect()
    }

    let a = generate_data(&mut rng);
    let b = generate_data(&mut rng);

    // Load the OCL kernel source code.
    let kernel = include_str!("../kernels/matmul.cl");

    // Compile the kernel into a runnable program.
    let pro_que = ProQue::builder()
        .src(kernel)
        .dims((N, N))
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
        .kernel_builder("matmul")
        .arg(N as i32)
        .arg(&a_buf)
        .arg(&b_buf)
        .arg(&dest_buf)
        .build()
        .expect("Failed to compile OpenCL kernel");

    unsafe {
        matmul.enq().expect("Failed to execute OpenCL kernel");
    }

    // Read back the result into `dest`.
    let mut dest = vec![0.0; SIZE];
    dest_buf.read(&mut dest).enq().unwrap();

    let (row, column) = (rng.gen_range(0..N), rng.gen_range(0..N));

    let result = dest[row * N + column];
    let mut correct_result = 0.0;
    for k in 0..N {
        correct_result += a[row * N + k] * b[k * N + column];
    }

    let correct = if result.approx_eq_ratio(&correct_result, 0.001) {
        '✓'
    } else {
        '❌'
    };

    println!(
        "(A[{}, :] @ B[:, {}]) = {} {}",
        row, column, result, correct
    );
}
