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

    let build_buffer = |data| {
        Buffer::<f32>::builder()
            .queue(pro_que.queue().clone())
            .dims(SIZE)
            .flags(MemFlags::new().read_only().copy_host_ptr())
            .host_data(data)
            .build().expect("Failed to create buffer")
    };

    let a_buf = build_buffer(&a);
    let b_buf = build_buffer(&b);

    let dest_buf = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .dims(SIZE)
        .flags(MemFlags::new().write_only())
        .build().expect("Failed to create destination buffer");

    let vadd = pro_que.create_kernel("vadd").expect("Failed to compile OpenCL kernel")
        .arg_buf(&a_buf)
        .arg_buf(&b_buf)
        .arg_buf(&dest_buf);

    unsafe {
        vadd.enq().expect("Failed to execute OpenCL kernel");
    }

    let mut dest = vec![0.0; SIZE];
    dest_buf.read(&mut dest).enq().unwrap();

    let index = thread_rng().gen_range(0, SIZE);
    let x = a[index];
    let y = b[index];
    let sum = dest[index];
    let good_sum = x + y;

    let correct = if sum.approx_eq_ratio(&good_sum, 0.001) { '✓' } else { '❌' };

    println!("{} + {} = {} {}", x, y, sum, correct);
}
