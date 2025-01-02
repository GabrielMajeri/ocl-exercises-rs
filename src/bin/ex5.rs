use ocl::flags::MemFlags;
use ocl::{Buffer, ProQue};

use rand::distributions::Standard;
use rand::{rngs::StdRng, Rng, SeedableRng};

use float_cmp::ApproxEqRatio;

const SEED: u64 = 42;
const SIZE: usize = 1024;

fn main() {
    println!("# Exercises 5 - Triple vector addition");
    println!(
        "Construct a kernel which directly adds three input vector into a single output vector"
    );
    vadd3();
}

pub fn vadd3() {
    // Generate some vectors containing random data.
    let mut rng = StdRng::seed_from_u64(SEED);

    fn generate_data(rng: &mut StdRng) -> Vec<f32> {
        rng.sample_iter(Standard).take(SIZE).collect()
    }

    let a = generate_data(&mut rng);
    let b = generate_data(&mut rng);
    let c = generate_data(&mut rng);

    // Load the OCL kernel source code.
    let kernel = include_str!("../kernels/vadd3.cl");

    // Compile the kernel into a runnable program.
    let pro_que = ProQue::builder()
        .src(kernel)
        .dims(SIZE)
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

    // We will have three input buffers,
    // each being a vector with randomly-initialized values.
    let a_buf = build_input_buffer(&a);
    let b_buf = build_input_buffer(&b);
    let c_buf = build_input_buffer(&c);

    let build_destination_buffer = || {
        Buffer::<f32>::builder()
            .queue(pro_que.queue().clone())
            .len(SIZE)
            .flags(MemFlags::new().write_only())
            .build()
            .expect("Failed to create destination buffer")
    };

    let dest_buf = build_destination_buffer();

    // Execute `dest_buf0 = a_buf + b_buf`
    let vadd3 = pro_que
        .kernel_builder("vadd3")
        .arg(&a_buf)
        .arg(&b_buf)
        .arg(&c_buf)
        .arg(&dest_buf)
        .build()
        .expect("Failed to compile OpenCL kernel");

    unsafe {
        vadd3.enq().expect("Failed to execute OpenCL kernel");
    }

    // Read back the result into `dest`.
    let mut dest = vec![0.0; SIZE];
    dest_buf.read(&mut dest).enq().unwrap();

    let index = rng.gen_range(0..SIZE);

    let x = a[index];
    let y = b[index];
    let z = c[index];

    let sum = dest[index];
    let good_sum = x + y + z;

    let correct = if sum.approx_eq_ratio(&good_sum, 0.001) {
        '✓'
    } else {
        '❌'
    };

    println!("{} + {} + {} = {} {}", x, y, z, sum, correct);
}
