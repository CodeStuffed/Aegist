//! The real GPU: NVIDIA's driver API, cuBLAS and NVRTC, loaded at run time
//! (so `aegist` still starts on machines without them). Only the handful
//! of C functions used here are declared; everything runs on the default
//! stream, so work happens in the order it's issued.

use super::{Arg, Backend, DevPtr, Gemm, Kernel, BLOCK, KERNELS_CU};
use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::PathBuf;

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type NvrtcProgram = *mut c_void;
type CublasHandle = *mut c_void;

const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
const CUBLAS_PEDANTIC_MATH: i32 = 2;
const CUDA_R_32F: i32 = 0;
const CUDA_R_16BF: i32 = 14;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_GEMM_DEFAULT: i32 = -1;
const CUBLAS_TF32_TENSOR_OP_MATH: i32 = 3;

struct Driver {
    init: unsafe extern "C" fn(u32) -> CuResult,
    device_get_count: unsafe extern "C" fn(*mut i32) -> CuResult,
    device_get: unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult,
    device_get_name: unsafe extern "C" fn(*mut c_char, i32, CuDevice) -> CuResult,
    device_get_attribute: unsafe extern "C" fn(*mut i32, i32, CuDevice) -> CuResult,
    device_total_mem: unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult,
    primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
    primary_ctx_release: unsafe extern "C" fn(CuDevice) -> CuResult,
    ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut DevPtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(DevPtr) -> CuResult,
    memcpy_htod: unsafe extern "C" fn(DevPtr, *const c_void, usize) -> CuResult,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, DevPtr, usize) -> CuResult,
    memset_d8: unsafe extern "C" fn(DevPtr, u8, usize) -> CuResult,
    module_load_data: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    module_get_function: unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    #[allow(clippy::type_complexity)]
    launch_kernel: unsafe extern "C" fn(CuFunction, u32, u32, u32, u32, u32, u32, u32, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> CuResult,
    get_error_string: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
    driver_get_version: unsafe extern "C" fn(*mut i32) -> CuResult,
}

struct Nvrtc {
    create_program: unsafe extern "C" fn(*mut NvrtcProgram, *const c_char, *const c_char, i32, *const *const c_char, *const *const c_char) -> i32,
    compile_program: unsafe extern "C" fn(NvrtcProgram, i32, *const *const c_char) -> i32,
    get_ptx_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> i32,
    get_ptx: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> i32,
    get_log_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> i32,
    get_log: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> i32,
    destroy_program: unsafe extern "C" fn(*mut NvrtcProgram) -> i32,
    version: unsafe extern "C" fn(*mut i32, *mut i32) -> i32,
}

struct Cublas {
    create: unsafe extern "C" fn(*mut CublasHandle) -> i32,
    destroy: unsafe extern "C" fn(CublasHandle) -> i32,
    set_math_mode: unsafe extern "C" fn(CublasHandle, i32) -> i32,
    #[allow(clippy::type_complexity)]
    sgemm_strided_batched: unsafe extern "C" fn(
        CublasHandle, i32, i32, i32, i32, i32, *const f32, *const f32, i32, i64, *const f32, i32, i64, *const f32, *mut f32, i32, i64, i32,
    ) -> i32,
    #[allow(clippy::type_complexity)]
    gemm_strided_batched_ex: unsafe extern "C" fn(
        CublasHandle, i32, i32, i32, i32, i32, *const c_void, *const c_void, i32, i32, i64, *const c_void, i32, i32, i64,
        *const c_void, *mut c_void, i32, i32, i64, i32, i32, i32,
    ) -> i32,
}

/// Places the CUDA libraries are usually installed, besides the system's search path.
fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(p) = std::env::var(var) {
            let p = PathBuf::from(p);
            dirs.extend([p.join("bin"), p.join("bin").join("x64"), p.join("lib64"), p.join("lib")]);
        }
    }
    if cfg!(windows) {
        if let Ok(root) = std::env::var("ProgramFiles") {
            let base = PathBuf::from(root).join("NVIDIA GPU Computing Toolkit").join("CUDA");
            let mut versions: Vec<PathBuf> = std::fs::read_dir(&base).map(|d| d.flatten().map(|e| e.path()).collect()).unwrap_or_default();
            versions.sort();
            for v in versions.into_iter().rev() {
                dirs.extend([v.join("bin"), v.join("bin").join("x64")]);
            }
        }
    } else {
        dirs.extend(["/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib/wsl/lib"].map(PathBuf::from));
    }
    dirs
}

/// Load a library by full path. On Windows its own folder is searched for
/// the libraries it loads in turn (NVRTC loads nvrtc-builtins from there),
/// even when that folder isn't on PATH.
fn load_path(path: &std::path::Path) -> Result<Library, libloading::Error> {
    #[cfg(windows)]
    {
        use libloading::os::windows::{Library as WinLibrary, LOAD_WITH_ALTERED_SEARCH_PATH};
        // Safety: as Library::new.
        unsafe { WinLibrary::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH) }.map(Library::from)
    }
    #[cfg(not(windows))]
    {
        // Safety: loading a library runs its initializers; these are NVIDIA's CUDA libraries.
        unsafe { Library::new(path) }
    }
}

fn open_library(what: &str, names: &[&str]) -> Result<Library> {
    let mut tried = Vec::new();
    for name in names {
        // the system's own search first, then the usual install folders
        // Safety: loading a library runs its initializers; these are NVIDIA's CUDA libraries.
        if let Ok(lib) = unsafe { Library::new(name) } {
            return Ok(lib);
        }
        for dir in search_dirs() {
            let path = dir.join(name);
            if path.is_file() {
                if let Ok(lib) = load_path(&path) {
                    return Ok(lib);
                }
            }
        }
        tried.push(*name);
    }
    bail!("couldn't load {what} (tried {})", tried.join(", "))
}

macro_rules! sym {
    ($lib:expr, $name:literal) => {
        // Safety: the declared type matches the C function's signature in NVIDIA's headers.
        *unsafe { $lib.get(concat!($name, "\0").as_bytes()) }.with_context(|| format!("{} is missing {}", stringify!($lib), $name))?
    };
}

pub struct CudaBackend {
    driver: Driver,
    cublas: Cublas,
    ctx: CuContext,
    device: CuDevice,
    handle: CublasHandle,
    functions: HashMap<Kernel, CuFunction>,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: u64,
    pub driver_version: i32,
    pub nvrtc_version: (i32, i32),
    _libs: Vec<Library>,
}

impl CudaBackend {
    /// Load the CUDA libraries, pick GPU 0 and compile the kernels.
    pub fn open() -> Result<Self> {
        let cublas_names: &[&str] = if cfg!(windows) {
            &["cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll"]
        } else {
            &["libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so"]
        };
        let (cuda_lib, driver) = load_driver()?;
        let info = device_info(&driver)?;
        let check = |what: &str, r: CuResult| driver_check(&driver, what, r);
        let device = info.device;
        let (name, cc, total, driver_version) = (info.name, info.compute_capability, info.total_memory, info.driver_version);
        let mut ctx: CuContext = std::ptr::null_mut();
        // Safety: out-pointer; device is valid.
        unsafe {
            check("cuDevicePrimaryCtxRetain", (driver.primary_ctx_retain)(&mut ctx, device))?;
            check("cuCtxSetCurrent", (driver.ctx_set_current)(ctx))?;
        }

        let (nvrtc_lib, nvrtc) = load_nvrtc()?;
        let mut nvrtc_version = (0, 0);
        // Safety: writes two ints.
        unsafe { (nvrtc.version)(&mut nvrtc_version.0, &mut nvrtc_version.1) };
        // Newest architecture first; an older NVRTC that doesn't know this
        // GPU still makes PTX the driver can compile for it.
        let archs = [format!("compute_{}{}", cc.0, cc.1), "compute_90".into(), "compute_80".into()];
        let mut ptx = None;
        let mut errors = Vec::new();
        for arch in &archs {
            match compile(&nvrtc, arch) {
                Ok(p) => {
                    ptx = Some(p);
                    break;
                }
                Err(e) => errors.push(format!("{arch}: {e}")),
            }
        }
        let ptx = ptx.ok_or_else(|| anyhow!("NVRTC couldn't compile the kernels:\n{}", errors.join("\n")))?;
        let mut module: CuModule = std::ptr::null_mut();
        // Safety: ptx is a NUL-terminated PTX string.
        check("cuModuleLoadData", unsafe { (driver.module_load_data)(&mut module, ptx.as_ptr() as *const c_void) })?;
        let mut functions = HashMap::new();
        for k in Kernel::ALL {
            let mut f: CuFunction = std::ptr::null_mut();
            let cname = CString::new(k.name()).unwrap();
            // Safety: valid module and NUL-terminated name.
            check(k.name(), unsafe { (driver.module_get_function)(&mut f, module, cname.as_ptr()) })?;
            functions.insert(k, f);
        }

        let cublas_lib = open_library("cuBLAS (part of the CUDA Toolkit)", cublas_names)
            .context("the CUDA Toolkit's cuBLAS library wasn't found - install the CUDA Toolkit 12.8 or newer")?;
        let cublas = Cublas {
            create: sym!(cublas_lib, "cublasCreate_v2"),
            destroy: sym!(cublas_lib, "cublasDestroy_v2"),
            set_math_mode: sym!(cublas_lib, "cublasSetMathMode"),
            sgemm_strided_batched: sym!(cublas_lib, "cublasSgemmStridedBatched"),
            gemm_strided_batched_ex: sym!(cublas_lib, "cublasGemmStridedBatchedEx"),
        };
        let mut handle: CublasHandle = std::ptr::null_mut();
        // Safety: out-pointer.
        let r = unsafe { (cublas.create)(&mut handle) };
        if r != 0 {
            bail!("cublasCreate failed (status {r})");
        }
        let be = CudaBackend {
            driver,
            cublas,
            ctx,
            device,
            handle,
            functions,
            name,
            compute_capability: cc,
            total_memory: total,
            driver_version,
            nvrtc_version,
            _libs: vec![cuda_lib, nvrtc_lib, cublas_lib],
        };
        be.set_tf32(true)?;
        Ok(be)
    }

    fn check(&self, what: &str, r: CuResult) -> Result<()> {
        driver_check(&self.driver, what, r)
    }

    /// Work may be issued from another thread than the one that opened the GPU.
    fn current(&self) -> Result<()> {
        // Safety: the context is alive for as long as self.
        self.check("cuCtxSetCurrent", unsafe { (self.driver.ctx_set_current)(self.ctx) })
    }
}

fn driver_check(driver: &Driver, what: &str, r: CuResult) -> Result<()> {
    if r == 0 {
        return Ok(());
    }
    let mut s: *const c_char = std::ptr::null();
    // Safety: cuGetErrorString only writes a pointer to a static string.
    let msg = unsafe {
        (driver.get_error_string)(r, &mut s);
        if s.is_null() { format!("error {r}") } else { CStr::from_ptr(s).to_string_lossy().into_owned() }
    };
    bail!("{what} failed: {msg}")
}

fn load_driver() -> Result<(Library, Driver)> {
    let names: &[&str] = if cfg!(windows) { &["nvcuda.dll"] } else { &["libcuda.so.1", "libcuda.so"] };
    let cuda_lib = open_library("the NVIDIA driver", names).context("no NVIDIA driver found")?;
    let driver = Driver {
        init: sym!(cuda_lib, "cuInit"),
        device_get_count: sym!(cuda_lib, "cuDeviceGetCount"),
        device_get: sym!(cuda_lib, "cuDeviceGet"),
        device_get_name: sym!(cuda_lib, "cuDeviceGetName"),
        device_get_attribute: sym!(cuda_lib, "cuDeviceGetAttribute"),
        device_total_mem: sym!(cuda_lib, "cuDeviceTotalMem_v2"),
        primary_ctx_retain: sym!(cuda_lib, "cuDevicePrimaryCtxRetain"),
        primary_ctx_release: sym!(cuda_lib, "cuDevicePrimaryCtxRelease_v2"),
        ctx_set_current: sym!(cuda_lib, "cuCtxSetCurrent"),
        ctx_synchronize: sym!(cuda_lib, "cuCtxSynchronize"),
        mem_get_info: sym!(cuda_lib, "cuMemGetInfo_v2"),
        mem_alloc: sym!(cuda_lib, "cuMemAlloc_v2"),
        mem_free: sym!(cuda_lib, "cuMemFree_v2"),
        memcpy_htod: sym!(cuda_lib, "cuMemcpyHtoD_v2"),
        memcpy_dtoh: sym!(cuda_lib, "cuMemcpyDtoH_v2"),
        memset_d8: sym!(cuda_lib, "cuMemsetD8_v2"),
        module_load_data: sym!(cuda_lib, "cuModuleLoadData"),
        module_get_function: sym!(cuda_lib, "cuModuleGetFunction"),
        launch_kernel: sym!(cuda_lib, "cuLaunchKernel"),
        get_error_string: sym!(cuda_lib, "cuGetErrorString"),
        driver_get_version: sym!(cuda_lib, "cuDriverGetVersion"),
    };
    Ok((cuda_lib, driver))
}

/// What the driver says about GPU 0.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    device: CuDevice,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: u64,
    /// e.g. 12080 for CUDA 12.8
    pub driver_version: i32,
}

fn device_info(driver: &Driver) -> Result<GpuInfo> {
    let check = |what: &str, r: CuResult| driver_check(driver, what, r);
    // Safety (this block): plain C calls with valid out-pointers.
    unsafe {
        check("cuInit", (driver.init)(0))?;
        let mut count = 0;
        check("cuDeviceGetCount", (driver.device_get_count)(&mut count))?;
        if count == 0 {
            bail!("the NVIDIA driver sees no GPU");
        }
        let mut device = 0;
        check("cuDeviceGet", (driver.device_get)(&mut device, 0))?;
        let mut buf = [0 as c_char; 256];
        check("cuDeviceGetName", (driver.device_get_name)(buf.as_mut_ptr(), 255, device))?;
        let name = CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned();
        let (mut major, mut minor) = (0, 0);
        check("cuDeviceGetAttribute", (driver.device_get_attribute)(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device))?;
        check("cuDeviceGetAttribute", (driver.device_get_attribute)(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device))?;
        let mut total = 0usize;
        check("cuDeviceTotalMem", (driver.device_total_mem)(&mut total, device))?;
        let mut v = 0;
        check("cuDriverGetVersion", (driver.driver_get_version)(&mut v))?;
        Ok(GpuInfo { device, name, compute_capability: (major, minor), total_memory: total as u64, driver_version: v })
    }
}

/// Look for an NVIDIA GPU without setting anything up (fast: for `doctor`).
pub fn probe() -> Result<GpuInfo> {
    let (_lib, driver) = load_driver()?;
    device_info(&driver)
}

fn load_nvrtc() -> Result<(Library, Nvrtc)> {
    let names: &[&str] = if cfg!(windows) {
        &["nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll"]
    } else {
        &["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so"]
    };
    let nvrtc_lib = open_library("NVRTC (part of the CUDA Toolkit)", names)
        .context("the CUDA Toolkit's NVRTC library wasn't found - install the CUDA Toolkit 12.8 or newer")?;
    let nvrtc = Nvrtc {
        create_program: sym!(nvrtc_lib, "nvrtcCreateProgram"),
        compile_program: sym!(nvrtc_lib, "nvrtcCompileProgram"),
        get_ptx_size: sym!(nvrtc_lib, "nvrtcGetPTXSize"),
        get_ptx: sym!(nvrtc_lib, "nvrtcGetPTX"),
        get_log_size: sym!(nvrtc_lib, "nvrtcGetProgramLogSize"),
        get_log: sym!(nvrtc_lib, "nvrtcGetProgramLog"),
        destroy_program: sym!(nvrtc_lib, "nvrtcDestroyProgram"),
        version: sym!(nvrtc_lib, "nvrtcVersion"),
    };
    Ok((nvrtc_lib, nvrtc))
}

/// The kernels as PTX for `arch` (e.g. "compute_120"), without needing a GPU.
pub fn compile_ptx(arch: &str) -> Result<String> {
    let (_lib, nvrtc) = load_nvrtc()?;
    Ok(compile(&nvrtc, arch)?.into_string()?)
}

fn compile(nvrtc: &Nvrtc, arch: &str) -> Result<CString> {
    let src = CString::new(KERNELS_CU).unwrap();
    let name = CString::new("aegist_kernels.cu").unwrap();
    let mut prog: NvrtcProgram = std::ptr::null_mut();
    // Safety (this block): NVRTC calls with valid pointers; the program is destroyed at the end.
    unsafe {
        let r = (nvrtc.create_program)(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null());
        if r != 0 {
            bail!("nvrtcCreateProgram failed ({r})");
        }
        let opts: Vec<CString> = [format!("--gpu-architecture={arch}"), "--std=c++17".into(), format!("-DBLOCK={BLOCK}")].into_iter().map(|o| CString::new(o).unwrap()).collect();
        let ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
        let r = (nvrtc.compile_program)(prog, ptrs.len() as i32, ptrs.as_ptr());
        let result = if r != 0 {
            let mut n = 0;
            (nvrtc.get_log_size)(prog, &mut n);
            let mut log = vec![0 as c_char; n.max(1)];
            (nvrtc.get_log)(prog, log.as_mut_ptr());
            Err(anyhow!("{}", CStr::from_ptr(log.as_ptr()).to_string_lossy()))
        } else {
            let mut n = 0;
            (nvrtc.get_ptx_size)(prog, &mut n);
            let mut ptx = vec![0u8; n];
            (nvrtc.get_ptx)(prog, ptx.as_mut_ptr() as *mut c_char);
            ptx.truncate(ptx.iter().position(|&b| b == 0).unwrap_or(ptx.len()));
            Ok(CString::new(ptx).unwrap())
        };
        (nvrtc.destroy_program)(&mut prog);
        result
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // Safety: handle and context were created in open() and are released once.
        unsafe {
            (self.cublas.destroy)(self.handle);
            (self.driver.primary_ctx_release)(self.device);
        }
    }
}

impl Backend for CudaBackend {
    fn describe(&self) -> String {
        format!("{} ({:.1} GB, compute capability {}.{}, driver {}.{}, NVRTC {}.{})",
            self.name, self.total_memory as f64 / (1u64 << 30) as f64, self.compute_capability.0, self.compute_capability.1,
            self.driver_version / 1000, self.driver_version % 1000 / 10, self.nvrtc_version.0, self.nvrtc_version.1)
    }

    fn alloc(&self, bytes: usize) -> Result<DevPtr> {
        self.current()?;
        let mut p = 0;
        // Safety: out-pointer.
        self.check(&format!("allocating {:.0} MB on the GPU", bytes as f64 / 1e6), unsafe { (self.driver.mem_alloc)(&mut p, bytes) })?;
        Ok(p)
    }

    fn free(&self, p: DevPtr) {
        let _ = self.current();
        // Safety: p came from alloc and is freed once.
        unsafe { (self.driver.mem_free)(p) };
    }

    fn upload(&self, dst: DevPtr, src: &[u8]) -> Result<()> {
        self.current()?;
        // Safety: src is valid for its length; dst was allocated at least that big.
        self.check("copying to the GPU", unsafe { (self.driver.memcpy_htod)(dst, src.as_ptr() as *const c_void, src.len()) })
    }

    fn download(&self, src: DevPtr, dst: &mut [u8]) -> Result<()> {
        self.current()?;
        // Safety: as upload.
        self.check("copying from the GPU", unsafe { (self.driver.memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, src, dst.len()) })
    }

    fn zero(&self, dst: DevPtr, bytes: usize) -> Result<()> {
        self.current()?;
        // Safety: dst was allocated at least `bytes` big.
        self.check("cuMemsetD8", unsafe { (self.driver.memset_d8)(dst, 0, bytes) })
    }

    fn launch(&self, k: Kernel, grid: (u32, u32), args: &[Arg]) -> Result<()> {
        self.current()?;
        let mut slots: Vec<u64> = args.iter().map(|a| a.slot()).collect();
        let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|s| s as *mut u64 as *mut c_void).collect();
        // Safety: one pointer per kernel parameter, each to a value of the parameter's size.
        let r = unsafe {
            (self.driver.launch_kernel)(self.functions[&k], grid.0, grid.1, 1, BLOCK, 1, 1, 0, std::ptr::null_mut(), ptrs.as_mut_ptr(), std::ptr::null_mut())
        };
        self.check(k.name(), r)
    }

    fn gemm(&self, g: &Gemm) -> Result<()> {
        self.current()?;
        let c = g.to_cublas();
        let (alpha, beta) = (c.alpha, c.beta);
        // Safety: pointers are device buffers sized for these shapes by the caller.
        let r = unsafe {
            (self.cublas.sgemm_strided_batched)(
                self.handle, c.trans_a as i32, c.trans_b as i32, c.m, c.n, c.k, &alpha,
                c.a as *const f32, c.lda, c.stride_a, c.b as *const f32, c.ldb, c.stride_b, &beta, c.c as *mut f32, c.ldc, c.stride_c, c.batch,
            )
        };
        if r != 0 {
            bail!("cuBLAS matrix multiply failed (status {r}) for {g:?}");
        }
        Ok(())
    }

    fn gemm_bf16(&self, g: &Gemm) -> Result<()> {
        self.current()?;
        let c = g.to_cublas();
        let (alpha, beta) = (c.alpha, c.beta);
        // Safety: A and B are bf16 device buffers, C f32, sized for these shapes by the caller.
        let r = unsafe {
            (self.cublas.gemm_strided_batched_ex)(
                self.handle, c.trans_a as i32, c.trans_b as i32, c.m, c.n, c.k, &alpha as *const f32 as *const c_void,
                c.a as *const c_void, CUDA_R_16BF, c.lda, c.stride_a, c.b as *const c_void, CUDA_R_16BF, c.ldb, c.stride_b,
                &beta as *const f32 as *const c_void, c.c as *mut c_void, CUDA_R_32F, c.ldc, c.stride_c, c.batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
            )
        };
        if r != 0 {
            bail!("cuBLAS bf16 matrix multiply failed (status {r}) for {g:?}");
        }
        Ok(())
    }

    fn supports_bf16(&self) -> bool {
        self.compute_capability.0 >= 8
    }

    fn sync(&self) -> Result<()> {
        self.current()?;
        // Safety: plain call.
        self.check("cuCtxSynchronize", unsafe { (self.driver.ctx_synchronize)() })
    }

    fn memory(&self) -> Result<(u64, u64)> {
        self.current()?;
        let (mut free, mut total) = (0usize, 0usize);
        // Safety: out-pointers.
        self.check("cuMemGetInfo", unsafe { (self.driver.mem_get_info)(&mut free, &mut total) })?;
        Ok((free as u64, total as u64))
    }

    fn set_tf32(&self, on: bool) -> Result<()> {
        // Safety: valid handle.
        let r = unsafe { (self.cublas.set_math_mode)(self.handle, if on { CUBLAS_TF32_TENSOR_OP_MATH } else { CUBLAS_PEDANTIC_MATH }) };
        if r != 0 {
            bail!("cublasSetMathMode failed (status {r})");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs where the CUDA Toolkit's NVRTC is installed (no GPU needed).
    #[test]
    fn kernels_compile_with_nvrtc_when_available() {
        if load_nvrtc().is_err() {
            eprintln!("NVRTC not installed; skipping");
            return;
        }
        for arch in ["compute_120", "compute_80"] {
            let ptx = compile_ptx(arch).unwrap();
            for k in Kernel::ALL {
                assert!(ptx.contains(&format!(".entry {}(", k.name())), "{arch}: {} missing", k.name());
            }
        }
        assert!(compile_ptx("compute_1").is_err());
    }

    #[test]
    fn no_gpu_is_a_clear_error_not_a_crash() {
        if let Err(e) = CudaBackend::open() {
            let msg = format!("{e:#}");
            assert!(msg.contains("NVIDIA") || msg.contains("GPU") || msg.contains("CUDA"), "{msg}");
        }
    }
}
