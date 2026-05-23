fn main() {
    let project_path = env!("CARGO_MANIFEST_DIR");

    println!("cargo:rustc-link-search=native={}/lib", project_path);
}
