//! Buffers and their memory: allocation, the resident weight cache and
//! the two upload paths (mapped, and through a staging buffer).

use super::Runtime;
use crate::device::VulkanError;
use ash::vk;
use frink_quant::fingerprint::WeightFingerprint;

/// One buffer and the memory bound to it, mapped when host-visible.
pub(super) struct Buf {
    pub(super) buffer: vk::Buffer,
    pub(super) memory: vk::DeviceMemory,
    pub(super) mapped: Option<*mut u8>,
    pub(super) size: usize,
}

pub(super) struct ResidentWeight {
    pub(super) buf: Buf,
    pub(super) fingerprint: WeightFingerprint,
}

impl Runtime {
    /// The device buffer holding `weights`, uploading it on a miss or
    /// when a different buffer now sits at the same address.
    pub(super) fn resident(&mut self, weights: &[u8]) -> Result<vk::Buffer, VulkanError> {
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

    pub(super) fn upload(&mut self, weights: &[u8]) -> Result<Buf, VulkanError> {
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
    pub(super) fn grow(
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
}

/// A buffer of `size` bytes with `usage`, in the first memory type that
/// has every flag in `wanted`; mapped when that type is host-visible.
pub(super) fn alloc(
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
pub(super) fn write_padded(buf: &Buf, bytes: &[u8]) {
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
pub(super) unsafe fn free(device: &ash::Device, buf: Buf) {
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
