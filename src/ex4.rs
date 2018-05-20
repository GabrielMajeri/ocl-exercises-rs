//! # Exercise 2
//!
//! Run a simple kernel which adds two vector, and stores the result in a third one.
//!
//! # Exercise 4
//!
//! Same as exercise 2, but using multiple vectors.
//!
//! This solves both exercise 2 and exercise 4.

use ocl::{ProQue, Buffer};
use ocl::flags::MemFlags;

use rand::{thread_rng, Rng};
use float_cmp::ApproxEqRatio;

const SIZE: usize = 1024;

pub fn vadd() {
    let kernel = include_str!("kernels/vadd.cl");

    let pro_que = ProQue::builder()
        .src(kernel)
        .dims(SIZE)
        .build()
        .expect("Failed to create OpenCL context");

    let generate_data = || {
        thread_rng().gen_iter::<f32>().take(SIZE).collect::<Vec<_>>()
    };

    let a = generate_data();
    let b = generate_data();
    let c = generate_data();

    let build_buffer = |data| {
        Buffer::<f32>::builder()
            .queue(pro_que.queue().clone())
            .len(SIZE)
            .flags(MemFlags::new().read_only().copy_host_ptr())
            .copy_host_slice(data)
            .build().expect("Failed to create buffer")
    };

    let a_buf = build_buffer(&a);
    let b_buf = build_buffer(&b);
    let c_buf = build_buffer(&c);

    let build_destination_buffer = || {
        Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(SIZE)
        .flags(MemFlags::new().write_only())
        .build().expect("Failed to create destination buffer")
    };

    let dest_buf0 = build_destination_buffer();

    let dest_buf1 = build_destination_buffer();

    let vadd = pro_que.kernel_builder("vadd")
        .arg(&a_buf)
        .arg(&b_buf)
        .arg(&dest_buf0)
        .build().expect("Failed to compile OpenCL kernel");

    unsafe {
        vadd.enq().expect("Failed to execute OpenCL kernel");
    }

    let vadd = pro_que.kernel_builder("vadd")
        .arg(&dest_buf0)
        .arg(&c_buf)
        .arg(&dest_buf1)
        .build().expect("Failed to compile OpenCL kernel");

    unsafe {
        vadd.enq().expect("Failed to execute OpenCL kernel");
    }

    let mut dest = vec![0.0; SIZE];
    dest_buf1.read(&mut dest).enq().unwrap();

    let index = thread_rng().gen_range(0, SIZE);

    let x = a[index];
    let y = b[index];
    let z = c[index];

    let sum = dest[index];
    let good_sum = x + y + z;

    let correct = if sum.approx_eq_ratio(&good_sum, 0.001) { '✓' } else { '❌' };

    println!("{} + {} + {} = {} {}", x, y, z, sum, correct);
}
