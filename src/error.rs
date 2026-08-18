use std::fmt;

/// Errors produced by vgfx.
#[derive(Debug)]
pub enum Error {
    /// No usable GPU device was found (e.g. no Metal device, no Vulkan driver).
    NoGpuDevice(String),
    /// The GPU could not create a pipeline (shader compile error, etc.).
    Pipeline(String),
    /// A surface/drawable could not be acquired.
    Surface(String),
    /// The backend is in an invalid state for the requested operation.
    Backend(String),
    /// Writing an image failed (bad path, encoding error).
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoGpuDevice(m) => write!(f, "no GPU device: {m}"),
            Error::Pipeline(m) => write!(f, "pipeline error: {m}"),
            Error::Surface(m) => write!(f, "surface error: {m}"),
            Error::Backend(m) => write!(f, "backend error: {m}"),
            Error::Io(m) => write!(f, "io error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

impl From<image::ImageError> for Error {
    fn from(e: image::ImageError) -> Self {
        Error::Io(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
