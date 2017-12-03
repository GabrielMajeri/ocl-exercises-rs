use ocl::ProQue;
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

    let a = thread_rng().gen_iter::<f32>().take(SIZE).collect::<Vec<_>>();
    let a_buf = pro_que.create_buffer::<f32>().unwrap();
    a_buf.write(&a).enq().unwrap();

    let b = thread_rng().gen_iter::<f32>().take(SIZE).collect::<Vec<_>>();
    let b_buf = pro_que.create_buffer::<f32>().unwrap();
    b_buf.write(&b).enq().unwrap();

    let dest_buf = pro_que.create_buffer::<f32>().unwrap();

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
