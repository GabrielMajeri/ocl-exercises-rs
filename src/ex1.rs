//! # Exercise 1 - Platform Information
//! Verify that running a simple OpenCL program works.

use ocl;

macro_rules! get_info {
    ($dev:ident, $name:ident) => {{
        use $crate::ocl::enums::{DeviceInfo, DeviceInfoResult};
        match $dev.info(DeviceInfo::$name) {
            DeviceInfoResult::$name(value) => value,
            _ => panic!("Failed to retrieve device {}", stringify!($name)),
        }
    }};
}

macro_rules! get_memory {
    ($dev:ident, $name:ident) => {{
        let memory = get_info!($dev, $name);
        memory_to_string(memory as usize)
    }};
}

/// Prints some information about the OpenCL environment.
pub fn opencl_info() {
    let platforms = ocl::Platform::list();

    println!("Number of OpenCL platforms: {}", platforms.len());

    for pl in platforms {
        println!(" * {}", pl.name());
        println!(" - Vendor: {}", pl.vendor());
        println!(" - Version: {}", pl.version());

        let devices = ocl::Device::list_all(pl).expect("Failed to list platform devices");

        println!(" - Device count: {}", devices.len());

        for dev in devices {
            // Some general information.
            println!("  * {}", dev.name());

            let version = dev.version().expect("Failed to retrieve device version");
            println!("  - Version: {}", version);

            // Information related to work-groups and work items.
            println!("  * Work-group information");

            let max_compute_units = get_info!(dev, MaxComputeUnits);
            println!("   - Maximum compute units: {}", max_compute_units);

            let max_wg_size = dev.max_wg_size().expect(
                "Failed to retrieve max work-group size",
            );
            println!("   - Maximum work-group total size: {}", max_wg_size);

            let max_wi_sizes = get_info!(dev, MaxWorkItemSizes);
            println!("   - Maximum work item dimensions: {:?}", max_wi_sizes);

            // Information related to memory and memory allocation.
            println!("  * Memory information");

            let local_mem_size = get_memory!(dev, LocalMemSize);
            println!("   - Local memory size: {}", local_mem_size);

            let global_mem_size = get_memory!(dev, GlobalMemSize);
            println!("   - Global memory size: {}", global_mem_size);

            let max_mem_alloc_size = get_memory!(dev, MaxMemAllocSize);
            println!(
                "   - Maximum memory allocation size: {}",
                max_mem_alloc_size
            );
        }
    }
}

/// Converts an amount of memory to a human-readable value.
fn memory_to_string(memory: usize) -> String {
    use number_prefix::{binary_prefix, Standalone, Prefixed};
    match binary_prefix(memory as f64) {
        Standalone(bytes) => format!("{} bytes", bytes),
        Prefixed(prefix, n) => format!("{:.0} {}B", n, prefix),
    }
}