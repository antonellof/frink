//! Run [`crate::q8_0_shader`] on a real device and read the answer back.
//!
//! # The claim this file makes, and what checks it
//!
//! Every `unsafe` block here is an `ash` FFI call. Individually they
//! are checked by their stated invariant; *collectively* they are
//! checked by [`crate::q8_0_reference::matvec_reference`], which
//! computes the same matvec on the host from the same bytes. The
//! module's own tests assert the two agree, which is the only thing
//! that can catch a descriptor bound to the wrong binding, a push
//! constant at the wrong offset, or a dispatch that covers the wrong
//! number of rows -- none of which are memory-unsafe and all of which
//! silently produce a wrong number.
//!
//! # Residency: staging, not zero-copy
//!
//! Weights are **copied** into host-visible device memory. The
//! zero-copy import the archived `amd-strix-halo` plan wants
//! (`VK_EXT_external_memory_host`, importing the GGUF mmap directly, as
//! `frink-metal`'s `register_weight_mmap` does with `BytesNoCopy`) is
//! deliberately out of scope here: it is an optional extension, it
//! needs page-aligned host pointers, and it is a property of a
//! *backend*, not of a GO/NO-GO. The verdict records it as unproven.

use crate::device::{Context, VulkanError};
use crate::q8_0_reference::pack_words;
use crate::q8_0_shader::{spirv, BLOCK_ELEMS, LOCAL_SIZE_X};
use ash::vk;

/// Bytes of push constants: `rows`, `n_blocks_per_row`, `row_bytes`.
const PUSH_CONSTANT_BYTES: u32 = 12;

/// One Q8_0 matvec on the GPU: `y[row] = dot(dequant(W[row]), x)`.
///
/// `weights` is `rows * row_bytes` of verbatim GGUF Q8_0 data --
/// exactly the slice `WeightMatrix::Quantized::data` hands the Metal
/// and CUDA launch functions. The argument list mirrors
/// `CudaMatvecLaunchFn` on purpose; see the seam notes in the verdict.
///
/// Builds and tears down the entire pipeline per call. That is correct
/// and absurdly wasteful, and it is why this function must never be
/// read as a performance path.
pub fn q8_0_matvec(
    ctx: &Context,
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, VulkanError> {
    assert_eq!(x.len(), n_blocks_per_row * BLOCK_ELEMS, "activation length");
    matvec(
        ctx,
        &spirv(),
        LOCAL_SIZE_X,
        weights,
        x,
        rows,
        row_bytes,
        n_blocks_per_row,
    )
}

/// The matvec for `kind` through its GLSL shader ([`crate::shaders`]),
/// with `q8_0_matvec`'s argument list so `frink-core` can hold each as a
/// plain function pointer.
macro_rules! shader_matvec {
    ($($fn_name:ident => $kind:literal),* $(,)?) => {$(
        #[doc = concat!("`", $kind, "` matvec through its GLSL shader.")]
        pub fn $fn_name(
            ctx: &Context,
            weights: &[u8],
            x: &[f32],
            rows: usize,
            row_bytes: usize,
            n_blocks_per_row: usize,
        ) -> Result<Vec<f32>, VulkanError> {
            let s = crate::shaders::matvec_shader($kind).expect("the table has the row");
            assert_eq!(x.len(), n_blocks_per_row * s.block_elems, "activation length");
            assert_eq!(row_bytes, n_blocks_per_row * s.block_bytes, "row stride");
            let code = crate::shaders::words(s.spv);
            matvec(ctx, &code, crate::shaders::LOCAL_SIZE_X, weights, x, rows, row_bytes, n_blocks_per_row)
        }
    )*};
}

shader_matvec! {
    q4_0_matvec => "Q4_0",
    q4_k_matvec => "Q4_K",
    q5_k_matvec => "Q5_K",
    q6_k_matvec => "Q6_K",
}

/// One matvec through any shader with the matvec interface: binding 0
/// the weight bytes as `uint` words, binding 1 the activations, binding
/// 2 the outputs, push constants `rows`, `n_blocks_per_row`, `row_bytes`,
/// one invocation per row in workgroups of `local_size_x`. Every kind's
/// shader (`crate::shaders`) keeps that interface, so the host side of a
/// new kind is a table row, not a copy of this function.
#[allow(clippy::too_many_arguments)] // the six matvec arguments plus the module and its width
pub fn matvec(
    ctx: &Context,
    code: &[u32],
    local_size_x: u32,
    weights: &[u8],
    x: &[f32],
    rows: usize,
    row_bytes: usize,
    n_blocks_per_row: usize,
) -> Result<Vec<f32>, VulkanError> {
    assert_eq!(weights.len(), rows * row_bytes, "weight buffer size");
    assert!(rows > 0 && n_blocks_per_row > 0, "empty matvec");

    let words = pack_words(weights);
    let device = &ctx.device;
    // SAFETY: `ctx.physical` belongs to `ctx.instance`, both live.
    let mem_props = unsafe {
        ctx.instance
            .get_physical_device_memory_properties(ctx.physical)
    };
    let mut owned = Owned::new(device);

    let w_buf = owned.storage_buffer(&mem_props, &bytes_of_u32(&words))?;
    let x_buf = owned.storage_buffer(&mem_props, &bytes_of_f32(x))?;
    let y_buf = owned.storage_buffer(&mem_props, &vec![0u8; rows * 4])?;

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
    owned.set_layout = Some(set_layout);

    let push_range = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(PUSH_CONSTANT_BYTES)];
    let set_layouts = [set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_range);
    // SAFETY: `layout_info` borrows two arrays alive across the call,
    // and `set_layout` was just created on this device.
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
        .map_err(VulkanError::vk("vkCreatePipelineLayout"))?;
    owned.pipeline_layout = Some(pipeline_layout);

    let module_info = vk::ShaderModuleCreateInfo::default().code(code);
    // SAFETY: `module_info` borrows `code`, alive across the call. The
    // words are a complete SPIR-V module (`spirv-val` asserts that in
    // `q8_0_shader`'s tests); an invalid one is rejected here rather
    // than being undefined behaviour.
    let module = unsafe { device.create_shader_module(&module_info, None) }
        .map_err(VulkanError::vk("vkCreateShaderModule"))?;
    owned.shader = Some(module);

    let entry = c"main";
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(entry);
    let pipeline_info = [vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout)];
    // SAFETY: `pipeline_info` borrows `stage`/`entry`, both alive; the
    // module and layout were created on this device above.
    let pipelines =
        unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None) }
            .map_err(|(_, code)| VulkanError::Vk {
                what: "vkCreateComputePipelines",
                code,
            })?;
    let pipeline = pipelines[0];
    owned.pipeline = Some(pipeline);

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(3)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    // SAFETY: `pool_info` borrows `pool_sizes`, alive across the call.
    let desc_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .map_err(VulkanError::vk("vkCreateDescriptorPool"))?;
    owned.desc_pool = Some(desc_pool);

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(desc_pool)
        .set_layouts(&set_layouts);
    // SAFETY: the pool was sized for exactly this one set of three
    // storage-buffer descriptors.
    let sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .map_err(VulkanError::vk("vkAllocateDescriptorSets"))?;
    let set = sets[0];

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
                .dst_set(set)
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        })
        .collect();
    // SAFETY: `writes` borrows `buffer_infos`, alive across the call;
    // every buffer is live and bound to memory, and the bindings match
    // the layout the pipeline was created with.
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family);
    // SAFETY: `ctx.queue_family` is the family this device was opened on.
    let cmd_pool = unsafe { device.create_command_pool(&pool_info, None) }
        .map_err(VulkanError::vk("vkCreateCommandPool"))?;
    owned.cmd_pool = Some(cmd_pool);

    let cmd_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: the pool was just created on this device.
    let cmd = unsafe { device.allocate_command_buffers(&cmd_info) }
        .map_err(VulkanError::vk("vkAllocateCommandBuffers"))?[0];

    let push = push_constants(rows, n_blocks_per_row, row_bytes);
    let groups = rows.div_ceil(local_size_x as usize) as u32;
    let begin =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    let barrier = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)];
    // SAFETY: `cmd` is a fresh primary buffer from `cmd_pool`, not
    // recorded into or submitted anywhere else. Every handle bound
    // below was created on `device` above and outlives the submission,
    // which is waited on before this function returns. The dispatch
    // covers `ceil(rows / 64)` groups of 64 invocations and the shader
    // guards `row < rows`, so no invocation writes past `y`.
    unsafe {
        device
            .begin_command_buffer(cmd, &begin)
            .map_err(VulkanError::vk("vkBeginCommandBuffer"))?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push,
        );
        device.cmd_dispatch(cmd, groups, 1, 1);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &barrier,
            &[],
            &[],
        );
        device
            .end_command_buffer(cmd)
            .map_err(VulkanError::vk("vkEndCommandBuffer"))?;
    }

    // SAFETY: a fresh unsignalled fence on this device.
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .map_err(VulkanError::vk("vkCreateFence"))?;
    owned.fence = Some(fence);
    let cmds = [cmd];
    let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
    // SAFETY: `submit` borrows `cmds`, alive across the call; `cmd` is
    // recorded and not submitted elsewhere; the wait immediately below
    // means every borrowed resource outlives GPU execution.
    unsafe {
        device
            .queue_submit(ctx.queue, &submit, fence)
            .map_err(VulkanError::vk("vkQueueSubmit"))?;
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(VulkanError::vk("vkWaitForFences"))?;
    }

    owned.read_f32(2, rows)
}

fn push_constants(rows: usize, n_blocks_per_row: usize, row_bytes: usize) -> [u8; 12] {
    let mut out = [0u8; 12];
    for (i, v) in [rows, n_blocks_per_row, row_bytes].iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&(*v as u32).to_le_bytes());
    }
    out
}

fn bytes_of_u32(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn bytes_of_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Everything created for one dispatch, destroyed in reverse on drop.
///
/// Exists so that an early `?` on any of the twenty fallible calls
/// above cannot leak a device object. Vulkan has no RAII of its own.
struct Owned<'a> {
    device: &'a ash::Device,
    buffers: Vec<vk::Buffer>,
    memories: Vec<vk::DeviceMemory>,
    mapped: Vec<*mut u8>,
    sizes: Vec<usize>,
    set_layout: Option<vk::DescriptorSetLayout>,
    pipeline_layout: Option<vk::PipelineLayout>,
    shader: Option<vk::ShaderModule>,
    pipeline: Option<vk::Pipeline>,
    desc_pool: Option<vk::DescriptorPool>,
    cmd_pool: Option<vk::CommandPool>,
    fence: Option<vk::Fence>,
}

impl<'a> Owned<'a> {
    fn new(device: &'a ash::Device) -> Self {
        Self {
            device,
            buffers: Vec::new(),
            memories: Vec::new(),
            mapped: Vec::new(),
            sizes: Vec::new(),
            set_layout: None,
            pipeline_layout: None,
            shader: None,
            pipeline: None,
            desc_pool: None,
            cmd_pool: None,
            fence: None,
        }
    }

    /// A `STORAGE_BUFFER` in HOST_VISIBLE|HOST_COHERENT memory,
    /// initialised from `contents` and left mapped.
    ///
    /// Host-coherent so no explicit flush or invalidate is needed:
    /// writes are visible to the device at queue submit, and device
    /// writes are visible to the host once the fence signals (with the
    /// HOST_READ barrier the command buffer records).
    fn storage_buffer(
        &mut self,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        contents: &[u8],
    ) -> Result<vk::Buffer, VulkanError> {
        let size = contents.len().max(4) as u64;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `info` is fully initialised and borrows nothing.
        let buffer = unsafe { self.device.create_buffer(&info, None) }
            .map_err(VulkanError::vk("vkCreateBuffer"))?;
        self.buffers.push(buffer);

        // SAFETY: `buffer` was just created on this device.
        let reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let type_index = find_memory_type(
            mem_props,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or(VulkanError::NoHostVisibleMemory)?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);
        // SAFETY: the size and type index come from this buffer's own
        // requirements and this device's memory properties.
        let memory = unsafe { self.device.allocate_memory(&alloc, None) }
            .map_err(VulkanError::vk("vkAllocateMemory"))?;
        self.memories.push(memory);
        // SAFETY: `memory` was allocated for `buffer`'s requirements and
        // is bound exactly once.
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(VulkanError::vk("vkBindBufferMemory"))?;

        // SAFETY: the memory is HOST_VISIBLE (checked above) and not
        // already mapped. The returned pointer is valid for `reqs.size`
        // bytes until `unmap_memory` in `drop`.
        let ptr = unsafe {
            self.device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(VulkanError::vk("vkMapMemory"))? as *mut u8;
        self.mapped.push(ptr);
        self.sizes.push(contents.len());

        // SAFETY: `ptr` is a live mapping of at least `reqs.size >=
        // contents.len()` bytes, `contents` is a distinct allocation, so
        // source and destination cannot overlap.
        unsafe { std::ptr::copy_nonoverlapping(contents.as_ptr(), ptr, contents.len()) };
        Ok(buffer)
    }

    /// Read `count` little-endian f32s back out of mapped buffer `idx`.
    ///
    /// Byte-wise rather than a `*const f32` read: `vkMapMemory` promises
    /// only `minMemoryMapAlignment`, and an unaligned `f32` load would
    /// be undefined behaviour on a target that cares.
    fn read_f32(&self, idx: usize, count: usize) -> Result<Vec<f32>, VulkanError> {
        let ptr = self.mapped[idx];
        let mut bytes = vec![0u8; count * 4];
        // SAFETY: buffer `idx` was allocated with at least `count * 4`
        // bytes (the caller sized it from the same `count`), the mapping
        // is live until `drop`, and the fence the caller waited on
        // ordered the device's writes before this read.
        unsafe { std::ptr::copy_nonoverlapping(ptr, bytes.as_mut_ptr(), count * 4) };
        let (words, rest) = bytes.as_chunks::<4>();
        debug_assert!(
            rest.is_empty(),
            "count * 4 is always a whole number of f32s"
        );
        Ok(words.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }
}

impl Drop for Owned<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle here was created on `self.device` and is
        // owned solely by this struct. The caller waits on its fence
        // before returning, so nothing is still executing. Each `take`
        // makes double-destruction impossible.
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(f) = self.fence.take() {
                self.device.destroy_fence(f, None);
            }
            if let Some(p) = self.cmd_pool.take() {
                self.device.destroy_command_pool(p, None);
            }
            if let Some(p) = self.desc_pool.take() {
                self.device.destroy_descriptor_pool(p, None);
            }
            if let Some(p) = self.pipeline.take() {
                self.device.destroy_pipeline(p, None);
            }
            if let Some(s) = self.shader.take() {
                self.device.destroy_shader_module(s, None);
            }
            if let Some(l) = self.pipeline_layout.take() {
                self.device.destroy_pipeline_layout(l, None);
            }
            if let Some(l) = self.set_layout.take() {
                self.device.destroy_descriptor_set_layout(l, None);
            }
            for b in self.buffers.drain(..) {
                self.device.destroy_buffer(b, None);
            }
            for m in self.memories.drain(..) {
                self.device.unmap_memory(m);
                self.device.free_memory(m, None);
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q8_0_reference::matvec_reference;
    use crate::q8_0_shader::BLOCK_BYTES;

    fn pseudo_random(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn push_constants_are_three_little_endian_u32s() {
        assert_eq!(
            push_constants(1, 2, 68),
            [1, 0, 0, 0, 2, 0, 0, 0, 68, 0, 0, 0]
        );
    }

    /// **The beachhead.** Runs the real shader on whatever Vulkan device
    /// this machine has and holds it against the scalar twin.
    ///
    /// Not `#[ignore]`d: it skips loudly when there is no device, so it
    /// is free on a driverless machine and automatic everywhere else.
    /// A tolerance is used rather than bit-equality because a GPU is
    /// free to contract `acc + a * b` into an FMA -- the same reason
    /// `frink-cuda`'s hardware test compares at 1e-4 relative.
    #[test]
    fn gpu_matvec_matches_the_scalar_twin() {
        let ctx = match Context::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIPPED, no Vulkan device: {e}");
                return;
            }
        };
        eprintln!("device: {}", ctx.device_name);

        // Shapes: one row/one block; a misaligned odd block count; a
        // row count that is not a multiple of the 64-wide workgroup;
        // and a shape wide enough to need several workgroups.
        for (rows, blocks) in [(1usize, 1usize), (9, 3), (65, 4), (200, 11)] {
            let cols = blocks * BLOCK_ELEMS;
            let row_bytes = blocks * BLOCK_BYTES;
            let mut weights = Vec::new();
            for r in 0..rows {
                weights.extend(frink_quant::quantize_q8_0(&pseudo_random(
                    0xbea4 + r as u64 * 7919,
                    cols,
                )));
            }
            let x = pseudo_random(0x515e, cols);
            let want = matvec_reference(&pack_words(&weights), &x, rows, row_bytes, blocks);
            let got = q8_0_matvec(&ctx, &weights, &x, rows, row_bytes, blocks)
                .expect("dispatch should succeed on a device that opened");
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let tol = 1e-4 * w.abs().max(1.0);
                assert!(
                    (g - w).abs() <= tol,
                    "rows={rows} blocks={blocks} row={i}: gpu {g} vs twin {w}"
                );
            }
        }
    }

    /// A row of `blocks` blocks of `kind`: deterministic bytes, with each
    /// block's f16 scale fields set to small finite values so no block
    /// decodes to inf or NaN.
    fn kind_row(kind: &str, blocks: usize, seed: u64) -> Vec<u8> {
        let shader = crate::shaders::matvec_shader(kind).unwrap();
        let mut s = seed | 1;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        };
        let mut out = Vec::new();
        for b in 0..blocks {
            let mut blk: Vec<u8> = (0..shader.block_bytes).map(|_| next()).collect();
            let d = half::f16::from_f32(0.01 + 0.002 * (b % 7) as f32).to_le_bytes();
            let dmin = half::f16::from_f32(0.005 + 0.001 * (b % 5) as f32).to_le_bytes();
            match kind {
                "Q4_0" => blk[0..2].copy_from_slice(&d),
                "Q4_K" | "Q5_K" => {
                    blk[0..2].copy_from_slice(&d);
                    blk[2..4].copy_from_slice(&dmin);
                }
                "Q6_K" => blk[208..210].copy_from_slice(&d),
                other => panic!("no layout for {other}"),
            }
            out.extend(blk);
        }
        out
    }

    fn twin(kind: &str) -> fn(&[u8], &[f32]) -> f32 {
        match kind {
            "Q4_0" => frink_quant::dot_q4_0_f32_scalar,
            "Q4_K" => frink_quant::dot_q4_k_f32_scalar,
            "Q5_K" => frink_quant::dot_q5_k_f32_scalar,
            "Q6_K" => frink_quant::dot_q6_k_f32_scalar,
            other => panic!("no twin for {other}"),
        }
    }

    /// **Every GLSL shader against its `frink_quant` scalar twin**, on
    /// whatever Vulkan device is present (MoltenVK on a Mac, RADV or
    /// AMDVLK on a Radeon), at shapes covering a row stride that is not
    /// word-aligned, a row count that is not a multiple of the 64-wide
    /// workgroup, and several workgroups. Skips loudly with no device.
    #[test]
    fn gpu_every_shader_matvec_matches_its_scalar_twin() {
        let ctx = match Context::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIPPED, no Vulkan device: {e}");
                return;
            }
        };
        type Launch =
            fn(&Context, &[u8], &[f32], usize, usize, usize) -> Result<Vec<f32>, VulkanError>;
        let launches: [(&str, Launch); 4] = [
            ("Q4_0", q4_0_matvec),
            ("Q4_K", q4_k_matvec),
            ("Q5_K", q5_k_matvec),
            ("Q6_K", q6_k_matvec),
        ];
        for (kind, launch) in launches {
            let shader = crate::shaders::matvec_shader(kind).unwrap();
            for (rows, blocks) in [(1usize, 1usize), (9, 3), (65, 2), (130, 5)] {
                let row_bytes = blocks * shader.block_bytes;
                let mut w = Vec::with_capacity(rows * row_bytes);
                for r in 0..rows {
                    w.extend(kind_row(kind, blocks, 31 * r as u64 + 7));
                }
                let x = pseudo_random(
                    rows as u64 * 17 + blocks as u64,
                    blocks * shader.block_elems,
                );
                let got = launch(&ctx, &w, &x, rows, row_bytes, blocks).expect("dispatch");
                for r in 0..rows {
                    let want = twin(kind)(&w[r * row_bytes..(r + 1) * row_bytes], &x);
                    let tol = 1e-4 * want.abs().max(1.0);
                    assert!(
                        (got[r] - want).abs() <= tol,
                        "{kind} ({rows},{blocks}) row {r}: gpu {} vs twin {want}",
                        got[r]
                    );
                }
            }
        }
    }
}
