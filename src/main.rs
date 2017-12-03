// https://docs.rs/ocl/
extern crate ocl;

extern crate number_prefix;
extern crate rand;
extern crate float_cmp;

mod ex1;
mod ex2;

fn main() {
    ex1::opencl_info();
    ex2::vadd();
}
