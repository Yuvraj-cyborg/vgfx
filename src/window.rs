//! Minimal macOS event loop for the bundled demos. No windowing dependency.
//! Create and run on the main thread. At most 64 native events are drained/frame.
use crate::backend::native::{Id, Object, Pool, class, msg, string};
use crate::{Error, Result};
use std::{
    cell::Cell,
    ffi::c_void,
    rc::{Rc, Weak},
};
#[link(name = "AppKit", kind = "framework")]
unsafe extern "C" {}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Point {
    x: f64,
    y: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Size {
    width: f64,
    height: f64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Rect {
    origin: Point,
    size: Size,
}
fn bounds(view: Id) -> Rect {
    #[cfg(target_arch = "aarch64")]
    {
        msg!(view,"bounds"=>Rect)
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        unsafe extern "C" {
            fn objc_msgSend_stret();
        }
        let f: unsafe extern "C" fn(*mut Rect, Id, Id) =
            std::mem::transmute(objc_msgSend_stret as unsafe extern "C" fn());
        let mut r = Rect::default();
        f(
            &mut r,
            view,
            crate::backend::native::sel_registerName(c"bounds".as_ptr()),
        );
        r
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ElementState {
    Pressed,
    Released,
}
pub enum MouseButton {
    Left,
}
#[derive(Clone, Copy)]
pub enum KeyCode {
    KeyW,
    KeyA,
    KeyS,
    KeyD,
    KeyR,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    ShiftLeft,
    ShiftRight,
    Space,
    Escape,
    Other,
}
pub enum PhysicalKey {
    Code(KeyCode),
}
pub struct KeyEvent {
    pub physical_key: PhysicalKey,
    pub state: ElementState,
}
pub struct Position {
    pub x: f64,
    pub y: f64,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PhysicalSize {
    pub width: u32,
    pub height: u32,
}
impl PhysicalSize {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}
impl From<PhysicalSize> for (u32, u32) {
    fn from(s: PhysicalSize) -> Self {
        (s.width, s.height)
    }
}
pub enum WindowEvent {
    CloseRequested,
    KeyboardInput {
        event: KeyEvent,
    },
    MouseInput {
        state: ElementState,
        button: MouseButton,
    },
    CursorMoved {
        position: Position,
    },
    Resized(PhysicalSize),
    RedrawRequested,
}
pub type WindowId = u64;
pub struct WindowAttributes {
    title: String,
    size: PhysicalSize,
}
impl Default for WindowAttributes {
    fn default() -> Self {
        Self {
            title: "vgfx".into(),
            size: PhysicalSize::new(960, 540),
        }
    }
}
impl WindowAttributes {
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = title.into();
        self
    }
    pub fn with_inner_size(mut self, size: PhysicalSize) -> Self {
        self.size = size;
        self
    }
}
struct State {
    window: Object,
    view: Id,
    redraw: Cell<bool>,
}
pub struct Window(Rc<State>);
impl Window {
    pub fn ns_view(&self) -> *mut c_void {
        self.0.view
    }
    pub fn inner_size(&self) -> PhysicalSize {
        let logical = bounds(self.0.view).size;
        let physical = msg!(self.0.view,"convertSizeToBacking:"=>Size,logical=>Size);
        PhysicalSize::new(
            physical.width.max(1.) as u32,
            physical.height.max(1.) as u32,
        )
    }
    pub fn request_redraw(&self) {
        self.0.redraw.set(true);
    }
    pub fn set_title(&self, title: &str) {
        let _pool = Pool::new();
        let title = string(title);
        msg!(self.0.window.id(),"setTitle:"=>(),title.id()=>Id);
    }
}
pub struct ActiveEventLoop {
    running: Cell<bool>,
    window: std::cell::RefCell<Weak<State>>,
}
impl ActiveEventLoop {
    pub fn exit(&self) {
        self.running.set(false);
    }
    pub fn create_window(&self, a: WindowAttributes) -> Result<Window> {
        let _pool = Pool::new();
        let obj = msg!(class(c"NSWindow"),"alloc"=>Id);
        let id = msg!(obj,"initWithContentRect:styleMask:backing:defer:"=>Id,Rect{origin:Point{x:100.,y:100.},size:Size{width:a.size.width as f64,height:a.size.height as f64}}=>Rect,15=>usize,2=>usize,false=>bool);
        let object = unsafe { Object::owned(id) }
            .ok_or_else(|| Error::Surface("NSWindow allocation".into()))?;
        msg!(id,"setReleasedWhenClosed:"=>(),false=>bool);
        let view = msg!(id,"contentView"=>Id);
        let state = Rc::new(State {
            window: object,
            view,
            redraw: Cell::new(true),
        });
        let w = Window(state);
        w.set_title(&a.title);
        msg!(id,"makeKeyAndOrderFront:"=>(),std::ptr::null_mut::<c_void>()=>Id);
        *self.window.borrow_mut() = Rc::downgrade(&w.0);
        Ok(w)
    }
}
pub trait ApplicationHandler {
    fn resumed(&mut self, event_loop: &ActiveEventLoop);
    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent);
}
pub struct EventLoop {
    application: Object,
}
impl EventLoop {
    pub fn new() -> Result<Self> {
        let _pool = Pool::new();
        let id = msg!(class(c"NSApplication"),"sharedApplication"=>Id);
        let application = unsafe { Object::retained(id) }
            .ok_or_else(|| Error::Surface("NSApplication".into()))?;
        msg!(id,"setActivationPolicy:"=>bool,0=>isize);
        msg!(id,"finishLaunching"=>());
        msg!(id,"activateIgnoringOtherApps:"=>(),true=>bool);
        Ok(Self { application })
    }
    pub fn run_app(&self, app: &mut impl ApplicationHandler) -> Result<()> {
        let active = ActiveEventLoop {
            running: Cell::new(true),
            window: std::cell::RefCell::new(Weak::new()),
        };
        app.resumed(&active);
        let mode = string("kCFRunLoopDefaultMode");
        let distant = unsafe { Object::retained(msg!(class(c"NSDate"),"distantPast"=>Id)) }
            .ok_or_else(|| Error::Surface("NSDate".into()))?;
        let mut last = None;
        let mut shift = false;
        while active.running.get() {
            let _pool = Pool::new();
            let Some(state) = active.window.borrow().upgrade() else {
                break;
            };
            for _ in 0..64 {
                let event = msg!(self.application.id(),"nextEventMatchingMask:untilDate:inMode:dequeue:"=>Id,usize::MAX=>usize,distant.id()=>Id,mode.id()=>Id,true=>bool);
                if event.is_null() {
                    break;
                }
                let kind = msg!(event,"type"=>usize);
                let translated = match kind {
                    1 | 2 => Some(WindowEvent::MouseInput {
                        state: if kind == 1 {
                            ElementState::Pressed
                        } else {
                            ElementState::Released
                        },
                        button: MouseButton::Left,
                    }),
                    5 | 6 => {
                        let p = msg!(event,"locationInWindow"=>Point);
                        let p = msg!(state.view,"convertPoint:fromView:"=>Point,p=>Point,std::ptr::null_mut::<c_void>()=>Id);
                        Some(WindowEvent::CursorMoved {
                            position: Position {
                                x: p.x,
                                y: bounds(state.view).size.height - p.y,
                            },
                        })
                    }
                    10 | 11 => {
                        let code = msg!(event,"keyCode"=>u16);
                        let key = match code {
                            0 => KeyCode::KeyA,
                            1 => KeyCode::KeyS,
                            2 => KeyCode::KeyD,
                            13 => KeyCode::KeyW,
                            15 => KeyCode::KeyR,
                            49 => KeyCode::Space,
                            53 => KeyCode::Escape,
                            123 => KeyCode::ArrowLeft,
                            124 => KeyCode::ArrowRight,
                            125 => KeyCode::ArrowDown,
                            126 => KeyCode::ArrowUp,
                            _ => KeyCode::Other,
                        };
                        Some(WindowEvent::KeyboardInput {
                            event: KeyEvent {
                                physical_key: PhysicalKey::Code(key),
                                state: if kind == 10 {
                                    ElementState::Pressed
                                } else {
                                    ElementState::Released
                                },
                            },
                        })
                    }
                    12 => {
                        let down = msg!(event,"modifierFlags"=>usize) & (1 << 17) != 0;
                        if down != shift {
                            shift = down;
                            Some(WindowEvent::KeyboardInput {
                                event: KeyEvent {
                                    physical_key: PhysicalKey::Code(KeyCode::ShiftLeft),
                                    state: if down {
                                        ElementState::Pressed
                                    } else {
                                        ElementState::Released
                                    },
                                },
                            })
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(event) = translated {
                    app.window_event(&active, 1, event);
                }
                msg!(self.application.id(),"sendEvent:"=>(),event=>Id);
            }
            if !msg!(state.window.id(),"isVisible"=>bool) {
                app.window_event(&active, 1, WindowEvent::CloseRequested);
                active.exit();
            }
            let w = Window(Rc::clone(&state));
            let size = w.inner_size();
            if last != Some(size) {
                last = Some(size);
                app.window_event(&active, 1, WindowEvent::Resized(size));
            }
            if active.running.get() && state.redraw.replace(false) {
                app.window_event(&active, 1, WindowEvent::RedrawRequested);
            }
            msg!(self.application.id(),"updateWindows"=>());
            if !state.redraw.get() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        Ok(())
    }
}
