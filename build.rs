// Only with `--features gpu-emulator`: compile the CPU emulator of the GPU
// kernels (src/gpu/emulate.cpp) so tests can run them without a GPU.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "gpu-emulator")]
    {
        println!("cargo:rerun-if-changed=src/gpu/emulate.cpp");
        println!("cargo:rerun-if-changed=src/gpu/kernels.cu");
        cc::Build::new()
            .cpp(true)
            .file("src/gpu/emulate.cpp")
            .flag_if_supported("-std=c++20")
            .flag_if_supported("/std:c++20")
            .flag_if_supported("-pthread")
            .flag_if_supported("-Wno-unused-parameter")
            .compile("aegist_gpu_emulator");
    }
}
