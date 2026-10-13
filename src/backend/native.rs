//! Minimal Objective-C ABI. Every call site specifies its exact method signature.
//! Owned objects release once; borrowed/autoreleased results must be retained.
use std::{
    ffi::{CStr, c_void},
    ptr::NonNull,
};
pub type Id = *mut c_void;
#[link(name = "objc")]
unsafe extern "C" {
    pub fn objc_getClass(name: *const i8) -> Id;
    pub fn sel_registerName(name: *const i8) -> Id;
    pub fn objc_msgSend();
    fn objc_autoreleasePoolPush() -> Id;
    fn objc_autoreleasePoolPop(pool: Id);
}
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}
#[link(name = "QuartzCore", kind = "framework")]
unsafe extern "C" {}
#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    pub fn MTLCreateSystemDefaultDevice() -> Id;
}
macro_rules! msg {
    ($obj:expr, $sel:literal => $ret:ty $(, $arg:expr => $ty:ty)* $(,)?) => {{
        // SAFETY: callers supply the SDK method signature, including scalar widths.
        #[allow(unused_unsafe)]
        unsafe {
            let f:unsafe extern "C" fn($crate::backend::native::Id,$crate::backend::native::Id $(,$ty)*)->$ret=
                std::mem::transmute($crate::backend::native::objc_msgSend as unsafe extern "C" fn());
            f($obj,$crate::backend::native::sel_registerName(concat!($sel,"\0").as_ptr().cast()) $(,$arg)*)
        }
    }};
}
pub(crate) use msg;
pub fn class(name: &CStr) -> Id {
    unsafe { objc_getClass(name.as_ptr()) }
}
pub struct Object(NonNull<c_void>);
impl Object {
    /// Adopt a +1 reference returned by a new/alloc/copy method.
    pub unsafe fn owned(id: Id) -> Option<Self> {
        NonNull::new(id).map(Self)
    }
    pub unsafe fn retained(id: Id) -> Option<Self> {
        if id.is_null() {
            None
        } else {
            unsafe { Self::owned(msg!(id,"retain"=>Id)) }
        }
    }
    pub fn id(&self) -> Id {
        self.0.as_ptr()
    }
    pub fn new(name: &CStr) -> Option<Self> {
        unsafe { Self::owned(msg!(class(name),"new"=>Id)) }
    }
}
impl Drop for Object {
    fn drop(&mut self) {
        msg!(self.id(),"release"=>());
    }
}
// Objects used by MetalBackend are exclusively accessed through &mut backend.
unsafe impl Send for Object {}
pub struct Pool(Id);
impl Pool {
    pub fn new() -> Self {
        Self(unsafe { objc_autoreleasePoolPush() })
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        unsafe { objc_autoreleasePoolPop(self.0) }
    }
}
pub fn string(text: &str) -> Object {
    let obj = msg!(class(c"NSString"),"alloc"=>Id);
    let obj = msg!(obj,"initWithBytes:length:encoding:"=>Id,text.as_ptr()=>*const u8,text.len()=>usize,4=>usize);
    unsafe { Object::owned(obj) }.expect("NSString allocation")
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}
