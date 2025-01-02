fn main() {
    println!("# Exercise 1 - Platform Information");
    println!("Verify that running a simple OpenCL program works");
    opencl_info();
}

/// Macro used to simplify the process for retrieving information
/// about a certain device.
macro_rules! get_info {
    ($dev:ident, $name:ident) => {{
        use ocl::enums::{DeviceInfo, DeviceInfoResult};
        match $dev.info(DeviceInfo::$name) {
            Ok(DeviceInfoResult::$name(value)) => value,
            _ => panic!("Failed to retrieve device {}", stringify!($name)),
        }
    }};
}

/// Macro used to retrieve information about the memory of a certain device.
macro_rules! get_memory {
    ($dev:ident, $name:ident) => {{
        let memory = get_info!($dev, $name);
        memory_to_string(memory as usize)
    }};
}

/// Prints some information about the OpenCL environment.
pub fn opencl_info() {
    // Get the list of available OpenCL platforms on this computer.
    let platforms = ocl::Platform::list();

    println!("Number of OpenCL platforms: {}", platforms.len());

    // Print information about each of them.
    platforms.iter().for_each(print_platform_info);
}

/// Prints some information about a given OpenCL platform.
fn print_platform_info(platform: &ocl::Platform) {
    println!(
        " * {}",
        platform.name().expect("Failed to retrieve platform name")
    );
    println!(
        " - Vendor: {}",
        platform
            .vendor()
            .expect("Failed to retrieve platform vendor")
    );
    println!(
        " - Version: {}",
        platform
            .version()
            .expect("Failed to retrieve platform verison")
    );

    // Retrieve all devices managed by this platform.
    let devices = ocl::Device::list_all(platform).expect("Failed to list platform devices");

    println!(" - Device count: {}", devices.len());

    devices.iter().for_each(print_device_info);
}

fn print_device_info(dev: &ocl::Device) {
    // Some general information.
    println!(
        "  * {}",
        dev.name().expect("Failed to retrieve device name")
    );

    let version = dev.version().expect("Failed to retrieve device version");
    println!("  - Version: {}", version);

    // Information related to work-groups and work items.
    println!("  * Work-group information");

    let max_compute_units = get_info!(dev, MaxComputeUnits);
    println!("   - Maximum compute units: {}", max_compute_units);

    let max_wg_size = dev
        .max_wg_size()
        .expect("Failed to retrieve max work-group size");
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

/// Converts an amount of memory to a human-readable value.
fn memory_to_string(memory: usize) -> String {
    use number_prefix::NumberPrefix;
    match NumberPrefix::binary(memory as f64) {
        NumberPrefix::Standalone(bytes) => format!("{} bytes", bytes),
        NumberPrefix::Prefixed(prefix, n) => format!("{:.0} {}B", n, prefix),
    }
}
