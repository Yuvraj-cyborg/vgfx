//! GPU backends: the platform-native one (Metal on Apple, Vulkan elsewhere)
//! behind a common [`Backend`] trait, plus the high-level [`Renderer`].

#[cfg(target_vendor = "apple")]
pub mod metal;
#[cfg(not(target_vendor = "apple"))]
pub mod vulkan;

pub mod common;

pub use common::{Backend, FrameTarget, Renderer};
