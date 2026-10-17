// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Low-Level OpenCL Zero-Copy UMA Bridge for Intel Integrated Graphics.
//!
//! Dynamically binds to OpenCL.dll on Windows (or libOpenCL.so on Unix) to allocate
//! `CL_MEM_USE_HOST_PTR` memory buffers directly over host `QuantumState` SOA arrays.
//! Executes parallel butterfly state evolutions on Intel Xe Graphics with zero PCIe copy.

use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

// OpenCL Type Aliases
pub type ClPlatformId = *mut c_void;
pub type ClDeviceId = *mut c_void;
pub type ClContext = *mut c_void;
pub type ClCommandQueue = *mut c_void;
pub type ClProgram = *mut c_void;
pub type ClKernel = *mut c_void;
pub type ClMem = *mut c_void;
pub type ClInt = i32;
pub type ClUint = u32;
pub type ClUlong = u64;

// OpenCL Constants
pub const CL_SUCCESS: ClInt = 0;
pub const CL_DEVICE_TYPE_GPU: ClUlong = 1 << 2;
pub const CL_PLATFORM_NAME: ClUint = 0x0902;
pub const CL_DEVICE_NAME: ClUint = 0x102B;
pub const CL_MEM_READ_WRITE: ClUlong = 1 << 0;
pub const CL_MEM_USE_HOST_PTR: ClUlong = 1 << 3;

#[cfg(target_os = "windows")]
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const c_char) -> *mut c_void;
    fn GetProcAddress(hModule: *mut c_void, lpProcName: *const c_char) -> *mut c_void;
    fn FreeLibrary(hLibModule: *mut c_void) -> i32;
}

pub const CL_TRUE: ClUint = 1;

/// Dynamically loaded OpenCL function dispatch table.
#[allow(non_snake_case, missing_debug_implementations)]
pub struct OpenClDispatch {
    _handle: *mut c_void,
    pub clGetPlatformIDs: unsafe extern "system" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt,
    pub clGetPlatformInfo: unsafe extern "system" fn(ClPlatformId, ClUint, usize, *mut c_void, *mut usize) -> ClInt,
    pub clGetDeviceIDs: unsafe extern "system" fn(ClPlatformId, ClUlong, ClUint, *mut ClDeviceId, *mut ClUint) -> ClInt,
    pub clGetDeviceInfo: unsafe extern "system" fn(ClDeviceId, ClUint, usize, *mut c_void, *mut usize) -> ClInt,
    pub clCreateContext: unsafe extern "system" fn(*const isize, ClUint, *const ClDeviceId, Option<extern "C" fn()>, *mut c_void, *mut ClInt) -> ClContext,
    pub clCreateCommandQueue: unsafe extern "system" fn(ClContext, ClDeviceId, ClUlong, *mut ClInt) -> ClCommandQueue,
    pub clCreateProgramWithSource: unsafe extern "system" fn(ClContext, ClUint, *const *const c_char, *const usize, *mut ClInt) -> ClProgram,
    pub clBuildProgram: unsafe extern "system" fn(ClProgram, ClUint, *const ClDeviceId, *const c_char, Option<extern "C" fn()>, *mut c_void) -> ClInt,
    pub clCreateKernel: unsafe extern "system" fn(ClProgram, *const c_char, *mut ClInt) -> ClKernel,
    pub clCreateBuffer: unsafe extern "system" fn(ClContext, ClUlong, usize, *mut c_void, *mut ClInt) -> ClMem,
    pub clSetKernelArg: unsafe extern "system" fn(ClKernel, ClUint, usize, *const c_void) -> ClInt,
    pub clEnqueueNDRangeKernel: unsafe extern "system" fn(ClCommandQueue, ClKernel, ClUint, *const usize, *const usize, *const usize, ClUint, *const c_void, *mut c_void) -> ClInt,
    pub clEnqueueReadBuffer: unsafe extern "system" fn(ClCommandQueue, ClMem, ClUint, usize, usize, *mut c_void, ClUint, *const c_void, *mut c_void) -> ClInt,
    pub clFinish: unsafe extern "system" fn(ClCommandQueue) -> ClInt,
    pub clReleaseMemObject: unsafe extern "system" fn(ClMem) -> ClInt,
    pub clReleaseKernel: unsafe extern "system" fn(ClKernel) -> ClInt,
    pub clReleaseProgram: unsafe extern "system" fn(ClProgram) -> ClInt,
    pub clReleaseCommandQueue: unsafe extern "system" fn(ClCommandQueue) -> ClInt,
    pub clReleaseContext: unsafe extern "system" fn(ClContext) -> ClInt,
}

impl OpenClDispatch {
    /// Attempts to dynamically load the OpenCL runtime library.
    pub fn load() -> Option<Self> {
        #[cfg(target_os = "windows")]
        unsafe {
            let lib_name = CString::new("OpenCL.dll").ok()?;
            let handle = LoadLibraryA(lib_name.as_ptr());
            if handle.is_null() {
                return None;
            }

            macro_rules! load_sym {
                ($sym:ident) => {{
                    let name = CString::new(stringify!($sym)).ok()?;
                    let ptr = GetProcAddress(handle, name.as_ptr());
                    if ptr.is_null() {
                        FreeLibrary(handle);
                        return None;
                    }
                    std::mem::transmute(ptr)
                }};
            }

            Some(Self {
                _handle: handle,
                clGetPlatformIDs: load_sym!(clGetPlatformIDs),
                clGetPlatformInfo: load_sym!(clGetPlatformInfo),
                clGetDeviceIDs: load_sym!(clGetDeviceIDs),
                clGetDeviceInfo: load_sym!(clGetDeviceInfo),
                clCreateContext: load_sym!(clCreateContext),
                clCreateCommandQueue: load_sym!(clCreateCommandQueue),
                clCreateProgramWithSource: load_sym!(clCreateProgramWithSource),
                clBuildProgram: load_sym!(clBuildProgram),
                clCreateKernel: load_sym!(clCreateKernel),
                clCreateBuffer: load_sym!(clCreateBuffer),
                clSetKernelArg: load_sym!(clSetKernelArg),
                clEnqueueNDRangeKernel: load_sym!(clEnqueueNDRangeKernel),
                clEnqueueReadBuffer: load_sym!(clEnqueueReadBuffer),
                clFinish: load_sym!(clFinish),
                clReleaseMemObject: load_sym!(clReleaseMemObject),
                clReleaseKernel: load_sym!(clReleaseKernel),
                clReleaseProgram: load_sym!(clReleaseProgram),
                clReleaseCommandQueue: load_sym!(clReleaseCommandQueue),
                clReleaseContext: load_sym!(clReleaseContext),
            })
        }
        #[cfg(not(target_os = "windows"))]
        {
            None
        }
    }
}

impl Drop for OpenClDispatch {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        if !self._handle.is_null() {
            unsafe {
                FreeLibrary(self._handle);
            }
        }
    }
}

/// OpenCL C source code for in-place quantum gate evolution.
pub const OPENCL_QUANTUM_SOURCE: &str = r#"
__kernel void ry_gate_kernel(
    __global double* real,
    __global double* imag,
    const ulong stride,
    const double cos_half,
    const double sin_half
) {
    size_t id = get_global_id(0);
    size_t k = id % stride;
    size_t block = (id / stride) * (stride * 2);
    size_t idx0 = block + k;
    size_t idx1 = idx0 + stride;

    double r0 = real[idx0];
    double i0 = imag[idx0];
    double r1 = real[idx1];
    double i1 = imag[idx1];

    real[idx0] = cos_half * r0 - sin_half * r1;
    imag[idx0] = cos_half * i0 - sin_half * i1;
    real[idx1] = sin_half * r0 + cos_half * r1;
    imag[idx1] = sin_half * i0 + cos_half * i1;
}

__kernel void rz_gate_kernel(
    __global double* real,
    __global double* imag,
    const ulong stride,
    const double cos_neg,
    const double sin_neg,
    const double cos_pos,
    const double sin_pos
) {
    size_t id = get_global_id(0);
    size_t k = id % stride;
    size_t block = (id / stride) * (stride * 2);
    size_t idx0 = block + k;
    size_t idx1 = idx0 + stride;

    double r0 = real[idx0];
    double i0 = imag[idx0];
    real[idx0] = r0 * cos_neg - i0 * sin_neg;
    imag[idx0] = r0 * sin_neg + i0 * cos_neg;

    double r1 = real[idx1];
    double i1 = imag[idx1];
    real[idx1] = r1 * cos_pos - i1 * sin_pos;
    imag[idx1] = r1 * sin_pos + i1 * cos_pos;
}

__kernel void cz_gate_kernel(
    __global double* real,
    __global double* imag,
    const ulong mask1,
    const ulong mask2
) {
    size_t idx = get_global_id(0);
    if ((idx & mask1) != 0 && (idx & mask2) != 0) {
        real[idx] = -real[idx];
        imag[idx] = -imag[idx];
    }
}
"#;

/// Zero-Copy OpenCL hardware context managing Intel Integrated Graphics.
#[allow(missing_debug_implementations)]
pub struct OpenClGpuContext {
    pub dispatch: OpenClDispatch,
    pub platform_name: String,
    pub device_name: String,
    pub context: ClContext,
    pub queue: ClCommandQueue,
    pub program: ClProgram,
    pub ry_kernel: ClKernel,
    pub rz_kernel: ClKernel,
    pub cz_kernel: ClKernel,
}

impl OpenClGpuContext {
    /// Attempts to initialize the Intel Graphics OpenCL context and compile quantum kernels.
    pub fn initialize() -> Option<Self> {
        let dispatch = OpenClDispatch::load()?;
        unsafe {
            let mut num_platforms = 0u32;
            let res = (dispatch.clGetPlatformIDs)(0, ptr::null_mut(), &mut num_platforms);
            if res != CL_SUCCESS || num_platforms == 0 {
                return None;
            }

            let mut platforms = vec![ptr::null_mut(); num_platforms as usize];
            (dispatch.clGetPlatformIDs)(num_platforms, platforms.as_mut_ptr(), ptr::null_mut());

            let mut _selected_platform = ptr::null_mut();
            let mut selected_device = ptr::null_mut();
            let mut platform_name_str = String::new();
            let mut device_name_str = String::new();

            for &p in &platforms {
                let mut buf = [0u8; 256];
                (dispatch.clGetPlatformInfo)(p, CL_PLATFORM_NAME, 256, buf.as_mut_ptr() as *mut c_void, ptr::null_mut());
                let p_name = CStr::from_ptr(buf.as_ptr() as *const c_char).to_string_lossy().to_string();

                let mut num_devices = 0u32;
                let d_res = (dispatch.clGetDeviceIDs)(p, CL_DEVICE_TYPE_GPU, 0, ptr::null_mut(), &mut num_devices);
                if d_res == CL_SUCCESS && num_devices > 0 {
                    let mut devices = vec![ptr::null_mut(); num_devices as usize];
                    (dispatch.clGetDeviceIDs)(p, CL_DEVICE_TYPE_GPU, num_devices, devices.as_mut_ptr(), ptr::null_mut());
                    _selected_platform = p;
                    selected_device = devices[0];
                    platform_name_str = p_name;

                    let mut d_buf = [0u8; 256];
                    (dispatch.clGetDeviceInfo)(selected_device, CL_DEVICE_NAME, 256, d_buf.as_mut_ptr() as *mut c_void, ptr::null_mut());
                    device_name_str = CStr::from_ptr(d_buf.as_ptr() as *const c_char).to_string_lossy().to_string();
                    break;
                }
            }

            if selected_device.is_null() {
                return None;
            }

            let mut err = 0i32;
            let context = (dispatch.clCreateContext)(ptr::null(), 1, &selected_device, None, ptr::null_mut(), &mut err);
            if err != CL_SUCCESS || context.is_null() {
                return None;
            }

            let queue = (dispatch.clCreateCommandQueue)(context, selected_device, 0, &mut err);
            if err != CL_SUCCESS || queue.is_null() {
                (dispatch.clReleaseContext)(context);
                return None;
            }

            let src_cstr = CString::new(OPENCL_QUANTUM_SOURCE).ok()?;
            let src_ptr = src_cstr.as_ptr();
            let program = (dispatch.clCreateProgramWithSource)(context, 1, &src_ptr, ptr::null(), &mut err);
            if err != CL_SUCCESS || program.is_null() {
                (dispatch.clReleaseCommandQueue)(queue);
                (dispatch.clReleaseContext)(context);
                return None;
            }

            let build_res = (dispatch.clBuildProgram)(program, 1, &selected_device, ptr::null(), None, ptr::null_mut());
            if build_res != CL_SUCCESS {
                (dispatch.clReleaseProgram)(program);
                (dispatch.clReleaseCommandQueue)(queue);
                (dispatch.clReleaseContext)(context);
                return None;
            }

            let k_ry = CString::new("ry_gate_kernel").ok()?;
            let k_rz = CString::new("rz_gate_kernel").ok()?;
            let k_cz = CString::new("cz_gate_kernel").ok()?;

            let ry_kernel = (dispatch.clCreateKernel)(program, k_ry.as_ptr(), &mut err);
            let rz_kernel = (dispatch.clCreateKernel)(program, k_rz.as_ptr(), &mut err);
            let cz_kernel = (dispatch.clCreateKernel)(program, k_cz.as_ptr(), &mut err);

            Some(Self {
                dispatch,
                platform_name: platform_name_str,
                device_name: device_name_str,
                context,
                queue,
                program,
                ry_kernel,
                rz_kernel,
                cz_kernel,
            })
        }
    }

    /// Executes zero-copy single-qubit Ry rotation on host statevector buffers.
    pub fn execute_ry_zero_copy(
        &self,
        real: &mut [f64],
        imag: &mut [f64],
        qubit: usize,
        theta: f64,
    ) -> Result<(), ClInt> {
        let dim = real.len();
        let stride = 1usize << qubit;
        let global_work_size = dim / 2;
        let (s, c) = (theta * 0.5).sin_cos();

        unsafe {
            let mut err = 0i32;
            let cl_real = (self.dispatch.clCreateBuffer)(
                self.context,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                dim * 8,
                real.as_mut_ptr() as *mut c_void,
                &mut err,
            );
            if err != CL_SUCCESS { return Err(err); }

            let cl_imag = (self.dispatch.clCreateBuffer)(
                self.context,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                dim * 8,
                imag.as_mut_ptr() as *mut c_void,
                &mut err,
            );
            if err != CL_SUCCESS {
                (self.dispatch.clReleaseMemObject)(cl_real);
                return Err(err);
            }

            (self.dispatch.clSetKernelArg)(self.ry_kernel, 0, std::mem::size_of::<ClMem>(), &cl_real as *const _ as *const c_void);
            (self.dispatch.clSetKernelArg)(self.ry_kernel, 1, std::mem::size_of::<ClMem>(), &cl_imag as *const _ as *const c_void);
            (self.dispatch.clSetKernelArg)(self.ry_kernel, 2, std::mem::size_of::<ClUlong>(), &stride as *const _ as *const c_void);
            (self.dispatch.clSetKernelArg)(self.ry_kernel, 3, std::mem::size_of::<f64>(), &c as *const _ as *const c_void);
            (self.dispatch.clSetKernelArg)(self.ry_kernel, 4, std::mem::size_of::<f64>(), &s as *const _ as *const c_void);

            (self.dispatch.clEnqueueNDRangeKernel)(
                self.queue,
                self.ry_kernel,
                1,
                ptr::null(),
                &global_work_size,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null_mut(),
            );

            (self.dispatch.clEnqueueReadBuffer)(
                self.queue,
                cl_real,
                CL_TRUE,
                0,
                dim * 8,
                real.as_mut_ptr() as *mut c_void,
                0,
                ptr::null(),
                ptr::null_mut(),
            );

            (self.dispatch.clEnqueueReadBuffer)(
                self.queue,
                cl_imag,
                CL_TRUE,
                0,
                dim * 8,
                imag.as_mut_ptr() as *mut c_void,
                0,
                ptr::null(),
                ptr::null_mut(),
            );

            (self.dispatch.clFinish)(self.queue);

            (self.dispatch.clReleaseMemObject)(cl_real);
            (self.dispatch.clReleaseMemObject)(cl_imag);
        }
        Ok(())
    }
}

impl Drop for OpenClGpuContext {
    fn drop(&mut self) {
        unsafe {
            (self.dispatch.clReleaseKernel)(self.ry_kernel);
            (self.dispatch.clReleaseKernel)(self.rz_kernel);
            (self.dispatch.clReleaseKernel)(self.cz_kernel);
            (self.dispatch.clReleaseProgram)(self.program);
            (self.dispatch.clReleaseCommandQueue)(self.queue);
            (self.dispatch.clReleaseContext)(self.context);
        }
    }
}
