//! Direct system Metal calls. Rust owns page-aligned shared buffer storage.
use super::native::{self, Id, Object, Pool, Size, msg};
use crate::{
    Error, Result,
    backend::common::{Draw, Frame, Gpu, U2D, make_u3d_with_projection},
    scene::{Camera, Mesh},
};
use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    ffi::c_void,
    ptr::NonNull,
    sync::Arc,
};

/// Owned CAMetalLayer. UI attachment must happen on the application's UI thread.
pub struct MetalLayer(Object);
/// # Safety
/// `view` must be a live NSView on its UI thread for the duration of this call.
pub unsafe fn attach_layer_to_view(
    view: *mut c_void,
    width: u32,
    height: u32,
) -> Result<MetalLayer> {
    if view.is_null() {
        return Err(Error::Surface("null NSView".into()));
    }
    let _pool = Pool::new();
    let device = unsafe { Object::owned(native::MTLCreateSystemDefaultDevice()) }
        .ok_or_else(|| Error::NoGpuDevice("no Metal device".into()))?;
    let layer = Object::new(c"CAMetalLayer")
        .ok_or_else(|| Error::Surface("CAMetalLayer allocation".into()))?;
    msg!(layer.id(),"setDevice:"=>(),device.id()=>Id);
    msg!(layer.id(),"setPixelFormat:"=>(),80=>usize);
    msg!(layer.id(),"setDrawableSize:"=>(),Size{width:width as f64,height:height as f64}=>Size);
    msg!(view,"setWantsLayer:"=>(),true=>bool);
    msg!(view,"setLayer:"=>(),layer.id()=>Id);
    Ok(MetalLayer(layer))
}
/// A no-copy MTLBuffer backed by a Rust allocation. The Metal object is dropped
/// before deallocating its bytes; submission is synchronous before reuse/drop.
struct Buffer {
    object: std::mem::ManuallyDrop<Object>,
    bytes: NonNull<u8>,
    layout: Layout,
}
unsafe impl Send for Buffer {}
impl Buffer {
    fn new(device: Id, length: usize) -> Result<Self> {
        let length = length
            .max(1)
            .checked_add(16383)
            .map(|n| n & !16383)
            .ok_or_else(|| Error::Backend("buffer size overflow".into()))?;
        let layout = Layout::from_size_align(length, 16384)
            .map_err(|_| Error::Backend("buffer layout".into()))?;
        let bytes = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| Error::Backend("buffer allocation".into()))?;
        let id = msg!(device,"newBufferWithBytesNoCopy:length:options:deallocator:"=>Id,bytes.as_ptr()=>*mut u8,length=>usize,0=>usize,std::ptr::null_mut::<c_void>()=>Id);
        let Some(object) = (unsafe { Object::owned(id) }) else {
            unsafe { dealloc(bytes.as_ptr(), layout) };
            return Err(Error::Backend("Metal shared buffer allocation".into()));
        };
        Ok(Self {
            object: std::mem::ManuallyDrop::new(object),
            bytes,
            layout,
        })
    }
    fn write<T: Copy>(&mut self, data: &[T]) {
        let length = std::mem::size_of_val(data);
        assert!(length <= self.layout.size());
        // Only padding-free GPU layouts and u32 indices are passed here.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), self.bytes.as_ptr(), length)
        };
    }
    fn id(&self) -> Id {
        self.object.id()
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.object);
            dealloc(self.bytes.as_ptr(), self.layout);
        }
    }
}
const SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct U2D {
    float4 vp;
};

typedef struct {
    packed_float2 position;
    packed_float4 color;
} V2D;

struct V2DOut {
    float4 position [[position]];
    float4 color;
};

vertex V2DOut v2d_vertex(
    const device V2D* verts [[buffer(0)]],
    unsigned int vid [[vertex_id]],
    constant U2D& u [[buffer(1)]])
{
    V2DOut out;
    // Screen space (y-down) -> NDC (y-up).
    float2 p = verts[vid].position;
    float2 ndc = float2(p.x / u.vp.x * 2.0 - 1.0, 1.0 - p.y / u.vp.y * 2.0);
    out.position = float4(ndc, 0.0, 1.0);
    out.color = float4(verts[vid].color);
    return out;
}

fragment float4 v2d_fragment(V2DOut in [[stage_in]]) {
    return in.color;
}

struct U3D {
    float4x4 mvp;
    float4x4 model;
    float3 cam_pos;
    float3 light_dir;
    float4 tint;
    float ambient;
    float specular;
    float _pad;
    float _pad2;
};

typedef struct {
    packed_float3 position;
    packed_float3 normal;
    packed_float4 color;
} V3D;

struct V3DOut {
    float4 position [[position]];
    float4 color;
    float3 normal;
    float3 world_pos;
};

vertex V3DOut v3d_vertex(
    const device V3D* verts [[buffer(0)]],
    unsigned int vid [[vertex_id]],
    constant U3D& u [[buffer(1)]])
{
    V3DOut out;
    float3 p = verts[vid].position;
    out.position = u.mvp * float4(p, 1.0);
    out.world_pos = (u.model * float4(p, 1.0)).xyz;
    // Exact for rigid / uniformly-scaled models; non-uniform scale would
    // need the inverse-transpose (not supported).
    out.normal = normalize((u.model * float4(verts[vid].normal, 0.0)).xyz);
    out.color = float4(verts[vid].color);
    return out;
}

fragment float4 v3d_fragment(V3DOut in [[stage_in]], constant U3D& u [[buffer(0)]]) {
    float3 n = normalize(in.normal);
    float3 l = normalize(u.light_dir);
    float3 v = normalize(u.cam_pos - in.world_pos);
    float diff = max(dot(n, l), 0.0);
    float3 hv = normalize(l + v);
    float spec = pow(max(dot(n, hv), 0.0), 48.0) * u.specular;
    float3 albedo = in.color.rgb * u.tint.rgb;
    float3 col = albedo * (u.ambient + (1.0 - u.ambient) * diff) + spec;
    return float4(col, in.color.a * u.tint.a);
}
"#;

struct MeshBuffers {
    mesh: Arc<Mesh>,
    vertices: Buffer,
    indices: Buffer,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Clear {
    r: f64,
    g: f64,
    b: f64,
    a: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Viewport {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    near: f64,
    far: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Origin {
    x: usize,
    y: usize,
    z: usize,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Extent {
    w: usize,
    h: usize,
    d: usize,
}

pub struct MetalBackend {
    device: Object,
    queue: Object,
    pipeline_2d: Object,
    pipeline_3d: Object,
    depth_state: Object,
    no_depth: Object,
    pass: Object,
    color: Object,
    depth: Object,
    layer: Option<MetalLayer>,
    width: u32,
    height: u32,
    vertices: Buffer,
    indices: Buffer,
    readback: Buffer,
    meshes: Vec<MeshBuffers>,
}
fn required(id: Id, what: &str) -> Result<Object> {
    unsafe { Object::owned(id) }.ok_or_else(|| Error::Backend(what.into()))
}
fn descriptor(class: &std::ffi::CStr) -> Result<Object> {
    Object::new(class).ok_or_else(|| Error::Backend("descriptor allocation".into()))
}
impl MetalBackend {
    fn texture(device: Id, w: u32, h: u32, format: usize) -> Result<Object> {
        let d = descriptor(c"MTLTextureDescriptor")?;
        msg!(d.id(),"setTextureType:"=>(),2=>usize);
        msg!(d.id(),"setWidth:"=>(),w as usize=>usize);
        msg!(d.id(),"setHeight:"=>(),h as usize=>usize);
        msg!(d.id(),"setPixelFormat:"=>(),format=>usize);
        msg!(d.id(),"setUsage:"=>(),4=>usize);
        msg!(d.id(),"setStorageMode:"=>(),2=>usize);
        required(
            msg!(device,"newTextureWithDescriptor:"=>Id,d.id()=>Id),
            "texture allocation",
        )
    }
    fn pipeline(device: Id, lib: Id, vert: &str, frag: &str, blended: bool) -> Result<Object> {
        let d = descriptor(c"MTLRenderPipelineDescriptor")?;
        let vert = native::string(vert);
        let frag = native::string(frag);
        let vf = required(
            msg!(lib,"newFunctionWithName:"=>Id,vert.id()=>Id),
            "vertex function",
        )?;
        let ff = required(
            msg!(lib,"newFunctionWithName:"=>Id,frag.id()=>Id),
            "fragment function",
        )?;
        msg!(d.id(),"setVertexFunction:"=>(),vf.id()=>Id);
        msg!(d.id(),"setFragmentFunction:"=>(),ff.id()=>Id);
        // Both pipelines participate in a pass with a depth attachment.
        msg!(d.id(),"setDepthAttachmentPixelFormat:"=>(),252=>usize);
        let attachments = msg!(d.id(),"colorAttachments"=>Id);
        let a = msg!(attachments,"objectAtIndexedSubscript:"=>Id,0=>usize);
        msg!(a,"setPixelFormat:"=>(),80=>usize);
        if blended {
            msg!(a,"setBlendingEnabled:"=>(),true=>bool);
            msg!(a,"setSourceRGBBlendFactor:"=>(),4=>usize);
            msg!(a,"setDestinationRGBBlendFactor:"=>(),5=>usize);
            msg!(a,"setSourceAlphaBlendFactor:"=>(),1=>usize);
            msg!(a,"setDestinationAlphaBlendFactor:"=>(),5=>usize);
        }
        let mut error: Id = std::ptr::null_mut();
        let id = msg!(device,"newRenderPipelineStateWithDescriptor:error:"=>Id,d.id()=>Id,&mut error=>*mut Id);
        if id.is_null() {
            return Err(Error::Pipeline(Self::error_message(error)));
        }
        required(id, "pipeline")
    }
    fn error_message(error: Id) -> String {
        if error.is_null() {
            return "native Metal call failed".into();
        }
        let text = msg!(error,"localizedDescription"=>Id);
        let bytes = msg!(text,"UTF8String"=>*const i8);
        if bytes.is_null() {
            return "native Metal error".into();
        }
        unsafe { std::ffi::CStr::from_ptr(bytes) }
            .to_string_lossy()
            .into_owned()
    }
    fn new(w: u32, h: u32, layer: Option<MetalLayer>) -> Result<Self> {
        if w == 0 || h == 0 {
            return Err(Error::Surface("zero-sized target".into()));
        }
        let _pool = Pool::new();
        let device = if let Some(l) = &layer {
            unsafe { Object::retained(msg!(l.0.id(),"device"=>Id)) }
        } else {
            unsafe { Object::owned(native::MTLCreateSystemDefaultDevice()) }
        }
        .ok_or_else(|| Error::NoGpuDevice("no Metal device".into()))?;
        let queue = required(msg!(device.id(),"newCommandQueue"=>Id), "command queue")?;
        let source = native::string(SHADERS);
        let mut error: Id = std::ptr::null_mut();
        let lib = msg!(device.id(),"newLibraryWithSource:options:error:"=>Id,source.id()=>Id,std::ptr::null_mut::<c_void>()=>Id,&mut error=>*mut Id);
        if lib.is_null() {
            return Err(Error::Pipeline(Self::error_message(error)));
        }
        let lib = required(lib, "library")?;
        let pipeline_2d =
            Self::pipeline(device.id(), lib.id(), "v2d_vertex", "v2d_fragment", true)?;
        let pipeline_3d =
            Self::pipeline(device.id(), lib.id(), "v3d_vertex", "v3d_fragment", false)?;
        let d = descriptor(c"MTLDepthStencilDescriptor")?;
        msg!(d.id(),"setDepthCompareFunction:"=>(),1=>usize);
        msg!(d.id(),"setDepthWriteEnabled:"=>(),true=>bool);
        let depth_state = required(
            msg!(device.id(),"newDepthStencilStateWithDescriptor:"=>Id,d.id()=>Id),
            "depth state",
        )?;
        msg!(d.id(),"setDepthCompareFunction:"=>(),7=>usize);
        msg!(d.id(),"setDepthWriteEnabled:"=>(),false=>bool);
        let no_depth = required(
            msg!(device.id(),"newDepthStencilStateWithDescriptor:"=>Id,d.id()=>Id),
            "depth state",
        )?;
        let color = Self::texture(device.id(), w, h, 80)?;
        let depth = Self::texture(device.id(), w, h, 252)?;
        let pass = descriptor(c"MTLRenderPassDescriptor")?;
        let vertices = Buffer::new(device.id(), 4 * 1024 * 1024)?;
        let indices = Buffer::new(device.id(), 1024 * 1024)?;
        let size = (w as usize)
            .checked_mul(h as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Surface("target size overflow".into()))?;
        let readback = Buffer::new(device.id(), size)?;
        Ok(Self {
            device,
            queue,
            pipeline_2d,
            pipeline_3d,
            depth_state,
            no_depth,
            pass,
            color,
            depth,
            layer,
            width: w,
            height: h,
            vertices,
            indices,
            readback,
            meshes: Vec::new(),
        })
    }
    pub fn offscreen(w: u32, h: u32) -> Result<Self> {
        Self::new(w, h, None)
    }
    pub fn window(w: u32, h: u32, layer: MetalLayer) -> Result<Self> {
        Self::new(w, h, Some(layer))
    }
    fn command(&self) -> Result<Object> {
        unsafe { Object::retained(msg!(self.queue.id(),"commandBuffer"=>Id)) }
            .ok_or_else(|| Error::Backend("command buffer".into()))
    }
    fn complete(cmd: Id) -> Result<()> {
        msg!(cmd,"commit"=>());
        msg!(cmd,"waitUntilCompleted"=>());
        if msg!(cmd,"status"=>usize) == 5 {
            return Err(Error::Backend(Self::error_message(msg!(cmd,"error"=>Id))));
        }
        Ok(())
    }
}
impl Gpu for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 || self.size() == (w, h) {
            return;
        }
        let _pool = Pool::new();
        let next = (|| -> Result<_> {
            Ok((
                Self::texture(self.device.id(), w, h, 80)?,
                Self::texture(self.device.id(), w, h, 252)?,
                Buffer::new(self.device.id(), w as usize * h as usize * 4)?,
            ))
        })();
        if let Ok((color, depth, readback)) = next {
            self.color = color;
            self.depth = depth;
            self.readback = readback;
            self.width = w;
            self.height = h;
            if let Some(l) = &self.layer {
                msg!(l.0.id(),"setDrawableSize:"=>(),Size{width:w as f64,height:h as f64}=>Size);
            }
        }
    }
    fn prepare_mesh(&mut self, mesh: &Arc<Mesh>) -> Result<usize> {
        if let Some(i) = self.meshes.iter().position(|m| Arc::ptr_eq(&m.mesh, mesh)) {
            return Ok(i);
        }
        if mesh
            .indices
            .iter()
            .any(|&i| i as usize >= mesh.vertices.len())
        {
            return Err(Error::Backend("mesh index out of bounds".into()));
        }
        // Retained assets have an explicit hard budget; no automatic eviction or
        // unbounded growth from accidentally rebuilding a mesh each frame.
        let used: usize = self
            .meshes
            .iter()
            .map(|m| m.vertices.layout.size() + m.indices.layout.size())
            .sum();
        let bytes = std::mem::size_of_val(mesh.vertices.as_slice())
            .checked_add(std::mem::size_of_val(mesh.indices.as_slice()))
            .and_then(|n| n.checked_add(32768));
        if self.meshes.len() >= 4096 || bytes.is_none_or(|n| n > 256 * 1024 * 1024 - used) {
            return Err(Error::Backend(
                "resident mesh budget exceeded (4096 slots / 256 MiB)".into(),
            ));
        }
        let mut vertices = Buffer::new(
            self.device.id(),
            std::mem::size_of_val(mesh.vertices.as_slice()),
        )?;
        let mut indices = Buffer::new(
            self.device.id(),
            std::mem::size_of_val(mesh.indices.as_slice()),
        )?;
        vertices.write(&mesh.vertices);
        indices.write(&mesh.indices);
        let i = self.meshes.len();
        self.meshes.push(MeshBuffers {
            mesh: Arc::clone(mesh),
            vertices,
            indices,
        });
        Ok(i)
    }
    fn reserve_geometry(&mut self, vertices: usize, indices: usize) -> Result<()> {
        let vb = vertices
            .checked_mul(std::mem::size_of::<super::common::V2D>())
            .ok_or_else(|| Error::Backend("vertex capacity overflow".into()))?;
        let ib = indices
            .checked_mul(4)
            .ok_or_else(|| Error::Backend("index capacity overflow".into()))?;
        if vb > self.vertices.layout.size() {
            self.vertices = Buffer::new(self.device.id(), vb)?;
        }
        if ib > self.indices.layout.size() {
            self.indices = Buffer::new(self.device.id(), ib)?;
        }
        Ok(())
    }
    fn render(&mut self, frame: &Frame, camera: &Camera) -> Result<()> {
        let _pool = Pool::new();
        let vb = std::mem::size_of_val(frame.vertices.as_slice());
        let ib = std::mem::size_of_val(frame.indices.as_slice());
        if vb > self.vertices.layout.size() || ib > self.indices.layout.size() {
            return Err(Error::Backend("frame exceeds reserved GPU geometry capacity; call reserve_geometry before recording".into()));
        }
        self.vertices.write(&frame.vertices);
        self.indices.write(&frame.indices);
        let drawable = if let Some(l) = &self.layer {
            Some(
                unsafe { Object::retained(msg!(l.0.id(),"nextDrawable"=>Id)) }
                    .ok_or_else(|| Error::Surface("no drawable".into()))?,
            )
        } else {
            None
        };
        let color = drawable
            .as_ref()
            .map(|d| msg!(d.id(),"texture"=>Id))
            .unwrap_or(self.color.id());
        let a = msg!(msg!(self.pass.id(),"colorAttachments"=>Id),"objectAtIndexedSubscript:"=>Id,0=>usize);
        msg!(a,"setTexture:"=>(),color=>Id);
        msg!(a,"setLoadAction:"=>(),2=>usize);
        msg!(a,"setStoreAction:"=>(),1=>usize);
        msg!(a,"setClearColor:"=>(),Clear{r:frame.clear.r as f64,g:frame.clear.g as f64,b:frame.clear.b as f64,a:frame.clear.a as f64}=>Clear);
        let d = msg!(self.pass.id(),"depthAttachment"=>Id);
        msg!(d,"setTexture:"=>(),self.depth.id()=>Id);
        msg!(d,"setLoadAction:"=>(),2=>usize);
        msg!(d,"setStoreAction:"=>(),0=>usize);
        msg!(d,"setClearDepth:"=>(),1.0=>f64);
        let cmd = self.command()?;
        let enc = msg!(cmd.id(),"renderCommandEncoderWithDescriptor:"=>Id,self.pass.id()=>Id);
        if enc.is_null() {
            return Err(Error::Backend("render encoder".into()));
        }
        msg!(enc,"setViewport:"=>(),Viewport{x:0.,y:0.,w:self.width as f64,h:self.height as f64,near:0.,far:1.}=>Viewport);
        let projection = camera.view_projection();
        let u2d = U2D {
            vp: [self.width as f32, self.height as f32, 0., 0.],
        };
        for draw in frame
            .draws
            .iter()
            .filter(|d| matches!(d, Draw::ThreeD { .. }))
            .chain(
                frame
                    .draws
                    .iter()
                    .filter(|d| matches!(d, Draw::TwoD { .. })),
            )
        {
            match draw {
                Draw::TwoD {
                    index_start,
                    index_count,
                } => {
                    if *index_count == 0 {
                        continue;
                    }
                    msg!(enc,"setRenderPipelineState:"=>(),self.pipeline_2d.id()=>Id);
                    msg!(enc,"setDepthStencilState:"=>(),self.no_depth.id()=>Id);
                    msg!(enc,"setVertexBuffer:offset:atIndex:"=>(),self.vertices.id()=>Id,0=>usize,0=>usize);
                    msg!(enc,"setVertexBytes:length:atIndex:"=>(),&u2d as *const U2D=>*const U2D,std::mem::size_of::<U2D>()=>usize,1=>usize);
                    msg!(enc,"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:"=>(),3=>usize,*index_count=>usize,1=>usize,self.indices.id()=>Id,*index_start*4=>usize);
                }
                Draw::ThreeD {
                    mesh,
                    model,
                    material,
                    gpu_slot,
                } => {
                    if mesh.is_empty() {
                        continue;
                    }
                    let Some(slot) = gpu_slot else {
                        return Err(Error::Backend("mesh was not prepared".into()));
                    };
                    let buffers = self
                        .meshes
                        .get(*slot)
                        .ok_or_else(|| Error::Backend("invalid mesh slot".into()))?;
                    let u = make_u3d_with_projection(camera, &projection, model, material);
                    msg!(enc,"setRenderPipelineState:"=>(),self.pipeline_3d.id()=>Id);
                    msg!(enc,"setDepthStencilState:"=>(),self.depth_state.id()=>Id);
                    msg!(enc,"setVertexBuffer:offset:atIndex:"=>(),buffers.vertices.id()=>Id,0=>usize,0=>usize);
                    msg!(enc,"setVertexBytes:length:atIndex:"=>(),&u as *const _=>*const super::common::U3D,std::mem::size_of_val(&u)=>usize,1=>usize);
                    msg!(enc,"setFragmentBytes:length:atIndex:"=>(),&u as *const _=>*const super::common::U3D,std::mem::size_of_val(&u)=>usize,0=>usize);
                    msg!(enc,"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:"=>(),3=>usize,mesh.indices.len()=>usize,1=>usize,buffers.indices.id()=>Id,0=>usize);
                }
            }
        }
        msg!(enc,"endEncoding"=>());
        if let Some(drawable) = &drawable {
            msg!(cmd.id(),"presentDrawable:"=>(),drawable.id()=>Id);
        }
        Self::complete(cmd.id())
    }
    fn readback(&mut self) -> Option<Vec<u8>> {
        if self.layer.is_some() {
            return None;
        }
        let _pool = Pool::new();
        let cmd = self.command().ok()?;
        let blit = msg!(cmd.id(),"blitCommandEncoder"=>Id);
        if blit.is_null() {
            return None;
        }
        let size = self.width as usize * self.height as usize * 4;
        msg!(blit,"copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:"=>(),self.color.id()=>Id,0=>usize,0=>usize,Origin{x:0,y:0,z:0}=>Origin,Extent{w:self.width as usize,h:self.height as usize,d:1}=>Extent,self.readback.id()=>Id,0=>usize,self.width as usize*4=>usize,size=>usize);
        msg!(blit,"endEncoding"=>());
        Self::complete(cmd.id()).ok()?;
        let mut rgba =
            unsafe { std::slice::from_raw_parts(self.readback.bytes.as_ptr(), size) }.to_vec();
        for px in rgba.as_chunks_mut::<4>().0 {
            px.swap(0, 2);
        }
        Some(rgba)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_matches_msl() {
        assert_eq!(std::mem::size_of::<super::super::common::V2D>(), 24);
        assert_eq!(std::mem::size_of::<crate::scene::Vertex3D>(), 40);
        assert_eq!(std::mem::size_of::<U2D>(), 16);
        assert_eq!(std::mem::size_of::<super::super::common::U3D>(), 192);
    }
}
