//! The Vulkan matvec runtime: everything that should outlive one call.
//!
//! The beachhead's `dispatch` built a set layout, pipeline layout,
//! shader module, pipeline, descriptor pool, command pool and fence,
//! and copied the whole weight matrix into fresh device memory, for
//! *every* matvec, then destroyed all of it. A Llama-3.2-1B Q4_K_M
//! decode through that ran at 0.9 tok/s on an M2 Pro against the CPU's
//! 99 (#409). [`Runtime`] keeps them:
//!
//! - **one** descriptor-set layout and pipeline layout, because every
//!   matvec shader has the same interface (`shaders/matvec_common.glsl`);
//! - **one pipeline per shader**, built on its first call;
//! - **one** command buffer, fence and descriptor set, re-recorded and
//!   re-written per call (the previous call waited on the fence, so
//!   none of them is in use);
//! - grow-only host-visible buffers for the activations and the output;
//! - **resident weights**, keyed by host `(address, length)` and checked
//!   against [`WeightFingerprint`] on every hit, the same identity rule
//!   `frink-cuda`'s cache uses. Production weights are one mmap for the
//!   life of the process, so each is uploaded once.
//!
//! # Where the weights live
//!
//! In `DEVICE_LOCAL` memory. When some `DEVICE_LOCAL` type is also
//! host-visible (unified memory: Apple, an APU such as Strix Halo, or a
//! discrete card with resizable BAR) the bytes are written through a
//! mapping. Otherwise, as on a discrete Radeon without ReBAR, they go
//! through a host-visible staging buffer and one `vkCmdCopyBuffer`.
//! [`Runtime::force_staging`] makes the second path testable on a
//! machine that would always take the first. Zero-copy import of the
//! mmap (`VK_EXT_external_memory_host`) is still not attempted.
//!
//! Nothing is evicted: like `frink-cuda`'s cache, the resident set is
//! every weight the model's matvecs have touched.

use crate::device::{Context, VulkanError};
use ash::vk;
use frink_quant::fingerprint::WeightFingerprint;
use std::collections::HashMap;

/// Bytes of push constants: `rows`, `n_blocks_per_row`, `row_bytes`.
const PUSH_CONSTANT_BYTES: u32 = 12;

/// One buffer and the memory bound to it, mapped when host-visible.
struct Buf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: Option<*mut u8>,
    size: usize,
}

struct ResidentWeight {
    buf: Buf,
    fingerprint: WeightFingerprint,
}

/// A compute pipeline and the module it was built from.
struct Kernel {
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

/// A Vulkan device with the matvec state kept across calls. See the
/// module docs.
pub struct Runtime {
    // `Drop for Runtime` destroys every handle below while the device
    // is alive; `ctx` is declared LAST so it drops last and closes it.
    mem_props: vk::PhysicalDeviceMemoryProperties,
    max_storage_range: u64,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    desc_pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    kernels: HashMap<&'static str, Kernel>,
    x: Option<Buf>,
    y: Option<Buf>,
    weights: HashMap<(usize, usize), ResidentWeight>,
    staging_only: bool,
    ctx: Context,
}

// SAFETY: the only non-`Send` fields are the raw mapped pointers in
// `Buf`, which point into device memory owned by this struct, freed
// only in its `Drop`, and are dereferenced only through `&mut self`.
// Moving the struct to another thread moves sole ownership of the
// mappings with it. Callers share it behind a `Mutex`, which also gives
// the queue the external synchronization Vulkan requires.
unsafe impl Send for Runtime {}

impl Runtime {
    /// Open the first compute-capable device and create the state every
    /// matvec shares.
    pub fn new() -> Result<Self, VulkanError> {
        let ctx = Context::new()?;
        let device = &ctx.device;
        // SAFETY: `ctx.physical` belongs to `ctx.instance`, both live.
        let mem_props = unsafe {
            ctx.instance
                .get_physical_device_memory_properties(ctx.physical)
        };
        // SAFETY: as above.
        let limits = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) }.limits;

        let bindings: Vec<_> = (0..3)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: `set_layout_info` borrows `bindings`, alive across the call.
        let set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }
            .map_err(VulkanError::vk("vkCreateDescriptorSetLayout"))?;

        // From here on every handle is owned by `rt`, so an early `?`
        // destroys what was made through `Drop for Runtime`; null
        // handles are skipped there.
        let mut rt = Self {
            mem_props,
            max_storage_range: u64::from(limits.max_storage_buffer_range),
            set_layout,
            pipeline_layout: vk::PipelineLayout::null(),
            desc_pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            cmd_pool: vk::CommandPool::null(),
            cmd: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            kernels: HashMap::new(),
            x: None,
            y: None,
            weights: HashMap::new(),
            staging_only: false,
            ctx,
        };
        let device = &rt.ctx.device;

        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_CONSTANT_BYTES)];
        let set_layouts = [set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_range);
        // SAFETY: `layout_info` borrows two arrays alive across the
        // call; `set_layout` was created on this device above.
        rt.pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .map_err(VulkanError::vk("vkCreatePipelineLayout"))?;

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        // SAFETY: `pool_info` borrows `pool_sizes`, alive across the call.
        rt.desc_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(VulkanError::vk("vkCreateDescriptorPool"))?;
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(rt.desc_pool)
            .set_layouts(&set_layouts);
        // SAFETY: the pool was sized for exactly this one set.
        rt.set = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .map_err(VulkanError::vk("vkAllocateDescriptorSets"))?[0];

        let cmd_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(rt.ctx.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: `queue_family` is the family this device was opened on.
        rt.cmd_pool = unsafe { device.create_command_pool(&cmd_pool_info, None) }
            .map_err(VulkanError::vk("vkCreateCommandPool"))?;
        let cmd_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(rt.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool was just created on this device.
        rt.cmd = unsafe { device.allocate_command_buffers(&cmd_info) }
            .map_err(VulkanError::vk("vkAllocateCommandBuffers"))?[0];
        // SAFETY: a fresh unsignalled fence on this device.
        rt.fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(VulkanError::vk("vkCreateFence"))?;
        Ok(rt)
    }

    /// The device this runtime opened.
    pub fn device_name(&self) -> &str {
        &self.ctx.device_name
    }

    /// Upload every new weight through a staging buffer even when the
    /// device has host-visible `DEVICE_LOCAL` memory: the path a
    /// discrete GPU without resizable BAR takes, made reachable on one
    /// that has it. Applies to weights uploaded after the call.
    pub fn force_staging(&mut self, on: bool) {
        self.staging_only = on;
    }

    /// How many distinct weight buffers are resident, and their bytes.
    pub fn resident_weights(&self) -> (usize, usize) {
        let bytes = self.weights.values().map(|w| w.buf.size).sum();
        (self.weights.len(), bytes)
    }

    /// One matvec: `y[row] = dot(dequant(W[row]), x)` through the
    /// shader `key`, built from `code()` on its first call. `weights`
    /// is `rows * row_bytes` of verbatim GGUF blocks and stays resident
    /// under its host address. Every shader passed here must have the
    /// matvec interface of `shaders/matvec_common.glsl`: one workgroup
    /// per row, rows folded over a 2-D grid.
    #[allow(clippy::too_many_arguments)] // the six matvec arguments plus the shader's identity
    pub fn matvec(
        &mut self,
        key: &'static str,
        code: impl FnOnce() -> Vec<u32>,
        weights: &[u8],
        x: &[f32],
        rows: usize,
        row_bytes: usize,
        n_blocks_per_row: usize,
    ) -> Result<Vec<f32>, VulkanError> {
        assert_eq!(weights.len(), rows * row_bytes, "weight buffer size");
        assert!(rows > 0 && n_blocks_per_row > 0, "empty matvec");
        let padded = weights.len().next_multiple_of(4) as u64;
        // The shader indexes bytes in a 32-bit `uint`, and a binding
        // cannot exceed the device's range: refuse rather than wrap.
        if padded > self.max_storage_range || padded > u64::from(u32::MAX) {
            return Err(VulkanError::TooLarge {
                bytes: padded,
                limit: self.max_storage_range.min(u64::from(u32::MAX)),
            });
        }

        let pipeline = self.kernel(key, code)?;
        let w_buf = self.resident(weights)?;
        let x_buf = Self::grow(&self.ctx.device, &self.mem_props, &mut self.x, x.len() * 4)?;
        let y_buf = Self::grow(&self.ctx.device, &self.mem_props, &mut self.y, rows * 4)?;
        let x_ptr = self
            .x
            .as_ref()
            .and_then(|b| b.mapped)
            .expect("host-visible");
        // SAFETY: `x_ptr` maps at least `x.len() * 4` bytes (`grow`
        // sized it), the previous call's fence wait means the device is
        // not reading it, and `x` is a distinct host allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr().cast::<u8>(), x_ptr, x.len() * 4);
        }

        let device = &self.ctx.device;
        let buffer_infos = [w_buf, x_buf, y_buf].map(|b| {
            [vk::DescriptorBufferInfo::default()
                .buffer(b)
                .offset(0)
                .range(vk::WHOLE_SIZE)]
        });
        let writes: Vec<_> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        // SAFETY: `writes` borrows `buffer_infos`, alive across the
        // call; every buffer is live and bound; the set is not in use
        // because every submission is waited on before returning.
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let push = push_constants(rows, n_blocks_per_row, row_bytes);
        let (gx, gy) = grid(rows);
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        let (layout, set) = (self.pipeline_layout, self.set);
        self.submit(|device, cmd| {
            // SAFETY: `cmd` is recording (see `submit`); every handle
            // was created on this device and outlives the submission,
            // which `submit` waits on. The grid has at least `rows`
            // workgroups and the shader guards `row < rows`, so no
            // invocation writes past `y`.
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[set],
                    &[],
                );
                device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
                device.cmd_dispatch(cmd, gx, gy, 1);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
            }
        })?;

        let y_ptr = self
            .y
            .as_ref()
            .and_then(|b| b.mapped)
            .expect("host-visible");
        let mut bytes = vec![0u8; rows * 4];
        // SAFETY: `y` maps at least `rows * 4` bytes, and the fence
        // `submit` waited on, with the HOST_READ barrier recorded above,
        // ordered the device's writes before this read. Byte-wise, as
        // `vkMapMemory` promises only `minMemoryMapAlignment`.
        unsafe { std::ptr::copy_nonoverlapping(y_ptr, bytes.as_mut_ptr(), rows * 4) };
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// The pipeline for `key`, built on first use.
    fn kernel(
        &mut self,
        key: &'static str,
        code: impl FnOnce() -> Vec<u32>,
    ) -> Result<vk::Pipeline, VulkanError> {
        if let Some(k) = self.kernels.get(key) {
            return Ok(k.pipeline);
        }
        let device = &self.ctx.device;
        let words = code();
        let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: `module_info` borrows `words`, alive across the call.
        // Every module is validated by `spirv-val` in its own crate's
        // tests; an invalid one is rejected here, not undefined.
        let module = unsafe { device.create_shader_module(&module_info, None) }
            .map_err(VulkanError::vk("vkCreateShaderModule"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layout)];
        // SAFETY: `info` borrows `stage`, alive; module and layout are
        // live handles on this device.
        let created =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &info, None) };
        let pipeline = match created {
            Ok(p) => p[0],
            Err((_, code)) => {
                // SAFETY: the module was created above and is unused.
                unsafe { device.destroy_shader_module(module, None) };
                return Err(VulkanError::Vk {
                    what: "vkCreateComputePipelines",
                    code,
                });
            }
        };
        self.kernels.insert(key, Kernel { module, pipeline });
        Ok(pipeline)
    }

    /// The device buffer holding `weights`, uploading it on a miss or
    /// when a different buffer now sits at the same address.
    fn resident(&mut self, weights: &[u8]) -> Result<vk::Buffer, VulkanError> {
        let key = (weights.as_ptr() as usize, weights.len());
        let fingerprint = WeightFingerprint::of(weights);
        if let Some(w) = self.weights.get(&key) {
            if w.fingerprint == fingerprint {
                return Ok(w.buf.buffer);
            }
            let stale = self.weights.remove(&key).expect("present");
            // SAFETY: every submission is waited on before returning,
            // so the stale buffer is not in use.
            unsafe { free(&self.ctx.device, stale.buf) };
        }
        let buf = self.upload(weights)?;
        let buffer = buf.buffer;
        self.weights
            .insert(key, ResidentWeight { buf, fingerprint });
        Ok(buffer)
    }

    fn upload(&mut self, weights: &[u8]) -> Result<Buf, VulkanError> {
        let size = weights.len().next_multiple_of(4);
        let device = &self.ctx.device;
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let mapped =
            local | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        if !self.staging_only {
            if let Ok(buf) = alloc(
                device,
                &self.mem_props,
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                mapped,
            ) {
                write_padded(&buf, weights);
                return Ok(buf);
            }
        }
        let dst = alloc(
            device,
            &self.mem_props,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            local,
        )?;
        let staging = match alloc(
            device,
            &self.mem_props,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(s) => s,
            Err(e) => {
                // SAFETY: `dst` was just made and never used.
                unsafe { free(device, dst) };
                return Err(e);
            }
        };
        write_padded(&staging, weights);
        let (src_buf, dst_buf) = (staging.buffer, dst.buffer);
        let region = [vk::BufferCopy::default().size(size as u64)];
        // Transfer writes made visible to every later compute read on
        // this queue: a barrier's second scope reaches later submissions.
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)];
        let copied = self.submit(|device, cmd| {
            // SAFETY: `cmd` is recording; both buffers are live, sized
            // `size`, with TRANSFER_SRC / TRANSFER_DST usage.
            unsafe {
                device.cmd_copy_buffer(cmd, src_buf, dst_buf, &region);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
            }
        });
        let device = &self.ctx.device;
        // SAFETY: `submit` waited on the copy (or failed before or
        // during submission and then waited idle), so staging is unused.
        unsafe { free(device, staging) };
        match copied {
            Ok(()) => Ok(dst),
            Err(e) => {
                // SAFETY: as above, for the destination.
                unsafe { free(device, dst) };
                Err(e)
            }
        }
    }

    /// Ensure `slot` is a host-visible storage buffer of at least
    /// `bytes`, replacing it when too small, and return it.
    fn grow(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        slot: &mut Option<Buf>,
        bytes: usize,
    ) -> Result<vk::Buffer, VulkanError> {
        if let Some(b) = slot {
            if b.size >= bytes {
                return Ok(b.buffer);
            }
        }
        if let Some(old) = slot.take() {
            // SAFETY: every submission is waited on before its call
            // returns, so the old buffer is idle.
            unsafe { free(device, old) };
        }
        let buf = alloc(
            device,
            mem_props,
            bytes.next_power_of_two().max(256),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let handle = buf.buffer;
        *slot = Some(buf);
        Ok(handle)
    }

    /// Record `body` into the one command buffer, submit it, and wait.
    fn submit(
        &mut self,
        body: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), VulkanError> {
        let device = &self.ctx.device;
        let cmd = self.cmd;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `cmd` came from a RESET_COMMAND_BUFFER pool and its
        // previous submission was waited on, so resetting it is legal.
        unsafe {
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(VulkanError::vk("vkResetCommandBuffer"))?;
            device
                .begin_command_buffer(cmd, &begin)
                .map_err(VulkanError::vk("vkBeginCommandBuffer"))?;
        }
        body(device, cmd);
        let cmds = [cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
        // SAFETY: `cmd` is fully recorded; the fence is unsignalled
        // (reset after every wait below); `submit` borrows `cmds`,
        // alive across the call; the queue is externally synchronized
        // by `&mut self`.
        let result = unsafe {
            device
                .end_command_buffer(cmd)
                .map_err(VulkanError::vk("vkEndCommandBuffer"))
                .and_then(|()| {
                    device
                        .queue_submit(self.ctx.queue, &submit, self.fence)
                        .map_err(VulkanError::vk("vkQueueSubmit"))
                })
                .and_then(|()| {
                    device
                        .wait_for_fences(&[self.fence], true, u64::MAX)
                        .map_err(VulkanError::vk("vkWaitForFences"))
                })
        };
        if result.is_err() {
            // SAFETY: leave nothing in flight behind an error, so the
            // callers' "every submission was waited on" holds.
            let _ = unsafe { device.device_wait_idle() };
        }
        // SAFETY: the fence is not in use by a pending submission.
        unsafe { device.reset_fences(&[self.fence]) }.map_err(VulkanError::vk("vkResetFences"))?;
        result
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let device = &self.ctx.device;
        // SAFETY: every handle was created on `device` and is owned
        // solely by this struct; the device is idled first; null handles
        // (from a `new` that failed part way) are skipped.
        unsafe {
            let _ = device.device_wait_idle();
            for (_, w) in self.weights.drain() {
                free(device, w.buf);
            }
            for b in [self.x.take(), self.y.take()].into_iter().flatten() {
                free(device, b);
            }
            for (_, k) in self.kernels.drain() {
                device.destroy_pipeline(k.pipeline, None);
                device.destroy_shader_module(k.module, None);
            }
            if self.fence != vk::Fence::null() {
                device.destroy_fence(self.fence, None);
            }
            if self.cmd_pool != vk::CommandPool::null() {
                device.destroy_command_pool(self.cmd_pool, None);
            }
            if self.desc_pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.desc_pool, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

/// A buffer of `size` bytes with `usage`, in the first memory type that
/// has every flag in `wanted`; mapped when that type is host-visible.
fn alloc(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: usize,
    usage: vk::BufferUsageFlags,
    wanted: vk::MemoryPropertyFlags,
) -> Result<Buf, VulkanError> {
    let info = vk::BufferCreateInfo::default()
        .size(size.max(4) as u64)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: `info` is fully initialised and borrows nothing.
    let buffer =
        unsafe { device.create_buffer(&info, None) }.map_err(VulkanError::vk("vkCreateBuffer"))?;
    // SAFETY: `buffer` was just created on this device.
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let Some(type_index) = find_memory_type(mem_props, reqs.memory_type_bits, wanted) else {
        // SAFETY: unbound and unused.
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(VulkanError::NoMemoryType {
            wanted: wanted.as_raw(),
        });
    };
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(type_index);
    // SAFETY: size and type come from this buffer's own requirements.
    let memory = match unsafe { device.allocate_memory(&alloc_info, None) } {
        Ok(m) => m,
        Err(code) => {
            // SAFETY: unbound and unused.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(VulkanError::Vk {
                what: "vkAllocateMemory",
                code,
            });
        }
    };
    let mut buf = Buf {
        buffer,
        memory,
        mapped: None,
        size,
    };
    // SAFETY: `memory` was allocated for `buffer`'s requirements and is
    // bound exactly once.
    if let Err(code) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        // SAFETY: never used by the device.
        unsafe { free(device, buf) };
        return Err(VulkanError::Vk {
            what: "vkBindBufferMemory",
            code,
        });
    }
    if wanted.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
        // SAFETY: the type is HOST_VISIBLE and the memory is not yet
        // mapped; the pointer stays valid until `free` unmaps it.
        match unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) } {
            Ok(p) => buf.mapped = Some(p.cast::<u8>()),
            Err(code) => {
                // SAFETY: never used by the device.
                unsafe { free(device, buf) };
                return Err(VulkanError::Vk {
                    what: "vkMapMemory",
                    code,
                });
            }
        }
    }
    Ok(buf)
}

/// Copy `bytes` into a mapped `buf` and zero its tail to the word.
fn write_padded(buf: &Buf, bytes: &[u8]) {
    let ptr = buf.mapped.expect("written buffers are host-visible");
    debug_assert!(buf.size >= bytes.len());
    // SAFETY: the mapping covers `buf.size >= bytes.len()` bytes (and
    // the zeroed tail stays inside `buf.size`); `bytes` is a distinct
    // host allocation and the device is not reading a buffer it was
    // never handed.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
        std::ptr::write_bytes(ptr.add(bytes.len()), 0, buf.size - bytes.len());
    }
}

/// Destroy a buffer and its memory.
///
/// # Safety
/// The device must not be using `buf`.
unsafe fn free(device: &ash::Device, buf: Buf) {
    // SAFETY: the caller guarantees `buf` is idle; it is consumed, so
    // it cannot be freed twice.
    unsafe {
        device.destroy_buffer(buf.buffer, None);
        if buf.mapped.is_some() {
            device.unmap_memory(buf.memory);
        }
        device.free_memory(buf.memory, None);
    }
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    wanted: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        type_bits & (1 << i) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(wanted)
    })
}

/// The workgroup grid for `rows` rows, one workgroup each: 1-D up to
/// 65535 (`maxComputeWorkGroupCount`'s guaranteed minimum), then 2-D,
/// which `matvec_row()` folds back (`y * NumWorkGroups.x + x`).
fn grid(rows: usize) -> (u32, u32) {
    let gx = rows.min(65535);
    (gx as u32, rows.div_ceil(gx) as u32)
}

pub(crate) fn push_constants(rows: usize, n_blocks_per_row: usize, row_bytes: usize) -> [u8; 12] {
    let mut out = [0u8; 12];
    for (i, v) in [rows, n_blocks_per_row, row_bytes].iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&(*v as u32).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_covers_every_row_and_stays_within_the_guaranteed_limit() {
        for rows in [1usize, 63, 65535, 65536, 128_256, 262_144] {
            let (gx, gy) = grid(rows);
            assert!(gx <= 65535 && gy <= 65535, "{rows}");
            assert!(gx as usize * gy as usize >= rows, "{rows}");
            assert!(
                (gx as usize) * (gy as usize - 1) < rows,
                "{rows}: a whole idle row of groups"
            );
        }
    }

    #[test]
    fn push_constants_are_three_little_endian_u32s() {
        assert_eq!(
            push_constants(1, 2, 68),
            [1, 0, 0, 0, 2, 0, 0, 0, 68, 0, 0, 0]
        );
    }
}
