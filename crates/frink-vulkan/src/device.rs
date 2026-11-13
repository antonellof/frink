//! Vulkan instance and device acquisition, via `ash`'s dynamic loader.
//!
//! Nothing here is frink-specific: it is the smallest sequence that
//! turns "there may be a driver on this machine" into "here is a device
//! with a compute queue", plus the two portability accommodations
//! MoltenVK needs on macOS.
//!
//! # Loading
//!
//! `ash`'s `loaded` feature opens `libvulkan` with `dlopen` at runtime,
//! so this crate builds with no Vulkan SDK, no headers and no linker
//! flags -- the same choice `frink-cuda` made with cudarc. The cost is
//! that the library has to be *findable*: Homebrew installs it in
//! `/opt/homebrew/lib`, which is not on the default macOS dlopen search
//! path, so [`load_entry`] tries a short list of known locations and
//! honours `FRINK_VULKAN_LOADER` for anything else.

use crate::select::DeviceKind;
use ash::vk;
use std::ffi::CStr;

/// Anything that can stop the beachhead reaching a device, named.
///
/// Every variant says what was missing rather than collapsing to "no
/// GPU": a refusal is coverage, and "you have a driver but no compute
/// queue" and "you have no driver" are different facts.
#[derive(Debug, thiserror::Error)]
pub enum VulkanError {
    #[error("no Vulkan loader found (tried {tried}); set FRINK_VULKAN_LOADER to a libvulkan path")]
    NoLoader { tried: String },
    #[error("vkCreateInstance failed: {0:?}")]
    InstanceCreation(vk::Result),
    #[error("the Vulkan loader reported no physical devices")]
    NoPhysicalDevice,
    /// Devices exist, but llama.cpp's rule leaves none of them visible:
    /// only CPU or virtual implementations (lavapipe, SwiftShader) and
    /// `GGML_VK_VISIBLE_DEVICES` unset.
    #[error(
        "no Vulkan GPU: the loader reports only {found}; set GGML_VK_VISIBLE_DEVICES to an index to use one"
    )]
    NoVisibleDevice { found: String },
    #[error(transparent)]
    Select(#[from] crate::select::SelectError),
    #[error("Vulkan device {0} exposes no queue family with COMPUTE")]
    NoComputeQueue(String),
    #[error("vkCreateDevice failed: {0:?}")]
    DeviceCreation(vk::Result),
    #[error("no memory type is both HOST_VISIBLE and HOST_COHERENT")]
    NoHostVisibleMemory,
    /// `wanted` is the raw `VkMemoryPropertyFlags` (ash's flag type has
    /// no `Debug` without its `debug` feature).
    #[error("no memory type has property flags {wanted:#x}")]
    NoMemoryType { wanted: u32 },
    #[error("a {bytes}-byte weight exceeds the {limit}-byte storage-buffer range")]
    TooLarge { bytes: u64, limit: u64 },
    /// A matvec whose weights, rows and activation do not describe whole
    /// blocks of its kind.
    #[error("shape rejected: {0}")]
    Shape(String),
    #[error("{what} failed: {code:?}")]
    Vk {
        what: &'static str,
        code: vk::Result,
    },
}

impl VulkanError {
    pub(crate) fn vk(what: &'static str) -> impl Fn(vk::Result) -> VulkanError {
        move |code| VulkanError::Vk { what, code }
    }
}

/// Paths tried before giving up, in order. `FRINK_VULKAN_LOADER` is
/// consulted first when set.
const LOADER_CANDIDATES: &[&str] = &[
    // Homebrew on Apple Silicon and on Intel macOS.
    "/opt/homebrew/lib/libvulkan.dylib",
    "/usr/local/lib/libvulkan.dylib",
    // Linux distributions.
    "libvulkan.so.1",
];

/// Open the Vulkan loader.
///
/// Tries `ash`'s own default first (which is correct on every platform
/// that installs the loader somewhere the dynamic linker looks), then
/// the explicit candidates.
pub fn load_entry() -> Result<ash::Entry, VulkanError> {
    let mut tried = Vec::new();
    if let Ok(path) = std::env::var("FRINK_VULKAN_LOADER") {
        // SAFETY: `Entry::load_from` is unsafe because it runs the
        // dynamic loader on a caller-supplied path and then trusts the
        // symbols it finds to be Vulkan's. The invariant is that the
        // path names a real Vulkan loader; the operator asserts that by
        // setting the variable, exactly as `LD_PRELOAD` is asserted.
        match unsafe { ash::Entry::load_from(&path) } {
            Ok(e) => return Ok(e),
            Err(_) => tried.push(path),
        }
    }
    // SAFETY: as above, for the platform's default library name.
    if let Ok(e) = unsafe { ash::Entry::load() } {
        return Ok(e);
    }
    tried.push("<platform default>".to_string());
    for candidate in LOADER_CANDIDATES {
        // SAFETY: as above, for a fixed allow-list of standard install
        // locations. A file at one of these paths that is not a Vulkan
        // loader fails symbol resolution and returns Err.
        match unsafe { ash::Entry::load_from(candidate) } {
            Ok(e) => return Ok(e),
            Err(_) => tried.push((*candidate).to_string()),
        }
    }
    Err(VulkanError::NoLoader {
        tried: tried.join(", "),
    })
}

/// A device with a compute queue, and the handles needed to free it.
///
/// Owns its `ash::Device`, `vk::Instance` and `ash::Entry`; [`Drop`]
/// destroys them in reverse order of creation.
pub struct Context {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    /// Human-readable device identity, for the verdict's receipt.
    pub device_name: String,
    /// `VK_API_VERSION` the physical device reports.
    pub api_version: u32,
}

impl Context {
    /// Create an instance and open `Vulkan0`: the first device of
    /// llama.cpp's visible list ([`crate::select::visible`]).
    pub fn new() -> Result<Self, VulkanError> {
        let entry = load_entry()?;
        let instance = create_instance(&entry)?;
        let picked = pick_vulkan0(&instance);
        let (physical, queue_family, device_name, api_version) = match picked {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: `instance` is live and not otherwise in use.
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };

        let device = match create_device(&instance, physical, queue_family) {
            Ok(d) => d,
            Err(e) => {
                // SAFETY: `instance` is live; the device was never made.
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };
        // SAFETY: `queue_family` index 0 was requested in
        // `create_device`, so this queue exists.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        Ok(Self {
            entry,
            instance,
            physical,
            device,
            queue,
            queue_family,
            device_name,
            api_version,
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: nothing else holds these handles (Context owns them
        // and is not Clone), and every command buffer submitted through
        // `dispatch` waits on its fence before returning, so the device
        // is idle here.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// One enumerated device, as `--list-devices` shows it.
#[derive(Debug, Clone)]
struct Enumerated {
    physical: vk::PhysicalDevice,
    name: String,
    kind: DeviceKind,
    api_version: u32,
}

fn device_kind(t: vk::PhysicalDeviceType) -> DeviceKind {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => DeviceKind::Discrete,
        vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceKind::Integrated,
        vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceKind::Virtual,
        vk::PhysicalDeviceType::CPU => DeviceKind::Cpu,
        _ => DeviceKind::Other,
    }
}

fn enumerate(instance: &ash::Instance) -> Result<Vec<Enumerated>, VulkanError> {
    // SAFETY: `instance` is live for the call; ash only reads from it.
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(VulkanError::vk("vkEnumeratePhysicalDevices"))?;
    Ok(devices
        .into_iter()
        .map(|physical| {
            // SAFETY: `physical` came from this instance's enumeration.
            let props = unsafe { instance.get_physical_device_properties(physical) };
            Enumerated {
                physical,
                name: cstr_name(&props.device_name),
                kind: device_kind(props.device_type),
                api_version: props.api_version,
            }
        })
        .collect())
}

/// The devices llama.cpp's rule makes visible, in `VulkanN` order.
fn visible_devices(instance: &ash::Instance) -> Result<Vec<Enumerated>, VulkanError> {
    let all = enumerate(instance)?;
    if all.is_empty() {
        return Err(VulkanError::NoPhysicalDevice);
    }
    let kinds: Vec<DeviceKind> = all.iter().map(|d| d.kind).collect();
    let env = std::env::var(crate::select::VISIBLE_ENV).ok();
    let picked = crate::select::visible(&kinds, env.as_deref())?;
    if picked.is_empty() {
        let found = all
            .iter()
            .map(|d| format!("{} ({})", d.name, d.kind.name()))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(VulkanError::NoVisibleDevice { found });
    }
    Ok(picked.into_iter().map(|i| all[i].clone()).collect())
}

fn pick_vulkan0(
    instance: &ash::Instance,
) -> Result<(vk::PhysicalDevice, u32, String, u32), VulkanError> {
    let first = visible_devices(instance)?.swap_remove(0);
    // SAFETY: `first.physical` came from this instance's enumeration.
    let families = unsafe { instance.get_physical_device_queue_family_properties(first.physical) };
    let family = families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .ok_or_else(|| VulkanError::NoComputeQueue(first.name.clone()))?;
    Ok((first.physical, family as u32, first.name, first.api_version))
}

/// One row of `--list-devices`: `Vulkan<index>` and what it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listed {
    pub index: usize,
    pub name: String,
    pub kind: DeviceKind,
    pub api_version: (u32, u32, u32),
}

/// Every visible device, in llama.cpp's `VulkanN` order, without
/// opening any of them. frink runs on index 0.
pub fn list() -> Result<Vec<Listed>, VulkanError> {
    let entry = load_entry()?;
    let instance = create_instance(&entry)?;
    let out = visible_devices(&instance);
    // SAFETY: `instance` is live, no device was created from it.
    unsafe { instance.destroy_instance(None) };
    Ok(out?
        .into_iter()
        .enumerate()
        .map(|(index, d)| Listed {
            index,
            name: d.name,
            kind: d.kind,
            api_version: (
                vk::api_version_major(d.api_version),
                vk::api_version_minor(d.api_version),
                vk::api_version_patch(d.api_version),
            ),
        })
        .collect())
}

fn create_instance(entry: &ash::Entry) -> Result<ash::Instance, VulkanError> {
    // SAFETY: reads the loader's static extension list; no handles.
    let available = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map_err(VulkanError::vk("vkEnumerateInstanceExtensionProperties"))?;
    let has = |name: &CStr| {
        available
            .iter()
            .any(|e| cstr_name(&e.extension_name) == name.to_string_lossy())
    };

    // MoltenVK is a *portability* driver: a Vulkan 1.0 loader hides it
    // unless the application opts in. Without these two the enumeration
    // above succeeds and returns zero devices, which reads exactly like
    // "no GPU" and is the single easiest way to conclude NO-GO by
    // mistake on a Mac.
    let mut extensions = Vec::new();
    let mut flags = vk::InstanceCreateFlags::empty();
    if has(ash::khr::portability_enumeration::NAME) {
        extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
        flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        if has(ash::khr::get_physical_device_properties2::NAME) {
            extensions.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }
    }

    let app_name = c"frink-vulkan-beachhead";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .api_version(vk::make_api_version(0, 1, 0, 0));
    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .flags(flags)
        .enabled_extension_names(&extensions);

    // SAFETY: `create_info` borrows `app_info` and `extensions`, both
    // alive until after this call returns, and names only extensions
    // the enumeration above reported as present.
    unsafe { entry.create_instance(&create_info, None) }.map_err(VulkanError::InstanceCreation)
}

fn create_device(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    queue_family: u32,
) -> Result<ash::Device, VulkanError> {
    // SAFETY: `physical` belongs to `instance`, which outlives the call.
    let available = unsafe { instance.enumerate_device_extension_properties(physical) }
        .map_err(VulkanError::vk("vkEnumerateDeviceExtensionProperties"))?;
    let mut extensions = Vec::new();
    // Required by the spec whenever the device advertises it: a
    // portability driver must be told the app knows it is one.
    if available.iter().any(|e| {
        cstr_name(&e.extension_name) == ash::khr::portability_subset::NAME.to_string_lossy()
    }) {
        extensions.push(ash::khr::portability_subset::NAME.as_ptr());
    }

    let priorities = [1.0f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities)];
    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&extensions);

    // SAFETY: `create_info` borrows `queue_info`/`priorities`/
    // `extensions`, all alive until the call returns; `queue_family`
    // came from this device's own family enumeration.
    unsafe { instance.create_device(physical, &create_info, None) }
        .map_err(VulkanError::DeviceCreation)
}

/// A `[c_char; N]` fixed-size Vulkan name field as a Rust `String`.
///
/// `c_char`, not `i8`: it is `u8` on aarch64 Linux, where an `&[i8]`
/// signature did not compile.
fn cstr_name(raw: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = raw
        .iter()
        .take_while(|c| **c != 0)
        .map(|c| *c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Whether this machine can run the beachhead at all, as one line.
///
/// Returns the device name on success. Used by the tests and by the
/// verdict so the answer is a receipt rather than a claim.
pub fn probe() -> Result<String, VulkanError> {
    let ctx = Context::new()?;
    Ok(format!(
        "{} (Vulkan {}.{}.{})",
        ctx.device_name,
        vk::api_version_major(ctx.api_version),
        vk::api_version_minor(ctx.api_version),
        vk::api_version_patch(ctx.api_version),
    ))
}
