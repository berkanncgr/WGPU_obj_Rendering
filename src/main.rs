use std::iter;

use cgmath::prelude::*;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use model::{DrawModel, Vertex};

mod model;
mod texture;

// Creates models as the square of the given number.
const NUM_INSTANCES_PER_ROW: u32 = 4;

/*
The coordinate system in Wgpu is based on DirectX, and Metal's coordinate systems.
That means that in normalized device coordinates the x axis and y axis are in the range of -1.0 to +1.0, and the z axis is 0.0 to +1.0.
The cgmath crate (as well as most game math crates) are built for OpenGL's coordinate system.
This matrix will scale and translate our scene from OpenGL's coordinate sytem to WGPU's. We'll define it as follows.

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);
*/

struct Camera
{
    eye:cgmath::Point3<f32>,     // Position
    target:cgmath::Point3<f32>,  // LookAt
    up:cgmath::Vector3<f32>,     // CameraUp
    aspect:f32,                  // ?View?
    fovy:f32,                    // Field of View
    znear:f32,                   // Near Plane
    zfar:f32,                    // Far Plane
}

impl Camera
{
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32>
    {   
        // The view matrix moves the world to be at the position and rotation of the camera.
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);

        // The proj matrix wraps the scene to give the effect of depth.
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        
        // Retrun Value:
        return proj * view
    }
}

#[repr(C)]  // We need this for Rust to store our data correctly for the shaders.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)] // This is so we can store this in a buffer.
struct CameraUniform
{   
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform
{
    fn new() -> Self
    {
        // Return value.
        Self
        {  
            view_position: [0.0; 4], 
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera)
    {
        // We're using Vector4 because ofthe camera_uniform 16 byte spacing requirement
        self.view_position = camera.eye.to_homogeneous().into();
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

struct CameraController
{
    speed: f32,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController
{
    fn new(speed: f32) -> Self
    {
        // Return value.
        Self
        {
            speed,
            is_up_pressed: false,
            is_down_pressed: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool
    {
        match event
        {
            WindowEvent::KeyboardInput
            {
                input:
                    KeyboardInput
                    {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => 
            {   
                // Make that key value "true" when a certain key pressed.
                let is_pressed = *state == ElementState::Pressed;
                match keycode
                {
                    VirtualKeyCode::Space =>
                    {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::LShift =>
                    {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::W | VirtualKeyCode::Up =>
                    {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left =>
                    {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right =>
                    {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn update_camera(&self, camera: &mut Camera)
    {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when camera gets too close to the center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed
        {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed
        {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the up/ down is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed
        {
            // Rescale the distance between the target and eye so
            // that it doesn't change. The eye therefore still
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed
        {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}

struct Instance
{
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

impl Instance
{
    fn to_raw(&self) -> InstanceRaw
    {
        InstanceRaw // Return value.
        {
            model: ( cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation) ).into(),
            normal: cgmath::Matrix3::from(self.rotation).into(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw
{
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],

    /* This is the data that will go into the wgpu::Buffer.
    We keep these separate so that we can update the Instance as much as we want without needing to mess with matrices. */
}

impl model::Vertex for InstanceRaw
{
    // Creates instances as many as square of given number from line 17.
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>
    {
        use std::mem;

        // A VertexBufferLayout defines how a buffer is layed out in memory.
        // Without this, the render_pipeline has no idea how to map the buffer in the shader.
        wgpu::VertexBufferLayout  // Return value.
        {
            // defines how wide a vertex is. When the shader goes to read the next vertex, it will skip over array_stride number of bytes.
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance.
            // This means that our shaders will only change to use the next instance when the shader starts processing a new instance.
            step_mode: wgpu::InputStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute
                {
                    offset: 0,
                    shader_location: 5, //  Tells the shader what location to store this attribute at.
                    format: wgpu::VertexFormat::Float32x4,
                },
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in the shader.
                wgpu::VertexAttribute
                {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute
                {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute
                {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute
                {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute
                {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute
                {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}



#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform
{
    position: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding: u32,
    color: [f32; 3],
}

// Everyting we use.
struct State
{
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,  // Make sure that our depth texture is the same size as our swap chain images.
    swap_chain: wgpu::SwapChain,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,  // for using shader.wgsl
    
    obj_model: model::Model, 

    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    instances: Vec<Instance>, // This parameter tells the GPU how many copies, or instances,
    instance_buffer: wgpu::Buffer,
    depth_texture: texture::Texture, // for depth testing

    light_bind_group: wgpu::BindGroup,
    light_render_pipeline: wgpu::RenderPipeline, // for using light.wgsl
}

// See the light in the scene.
fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline
{
    let shader = device.create_shader_module(&shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor
    {
        label: Some("Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState
        {
            module: &shader,
            entry_point: "main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState
        {
            module: &shader,
            entry_point: "main",
            targets: &[wgpu::ColorTargetState
            {
                format: color_format,
                blend: Some(wgpu::BlendState
                {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrite::ALL,
            }],
        }),
        primitive: wgpu::PrimitiveState
        {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLAMPING
            clamp_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState
        {
            format,
            depth_write_enabled: true,
            // Tells us when to discard a new pixel. Using LESS means pixels will be drawn front to back.
            depth_compare: wgpu::CompareFunction::Less, 
            stencil: wgpu::StencilState::default(), // Control values for stencil testing.
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState
        {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
    })
}

impl State
{   
    // Creating some of the wgpu types requires async code
    async fn new(window: &Window) -> Self  // Returns State struct.
    { 
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions
            {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
            }
        ).await.unwrap();

        // We need the adapter to create the device and queue.
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor
        {
            // DeviceDescriptor, allows us to specify what extra features we want.
            label: None,
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
        },    
        None, // Trace path
        ).await.unwrap();

        let sc_desc = wgpu::SwapChainDescriptor
        {
            // usage describes how the swap_chain's underlying textures will be used.
            // RENDER_ATTACHMENT specifies that the textures will be used to write to the screen.
            // The format defines how the swap_chains textures will be stored on the gpu.
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            format: adapter.get_swap_chain_preferred_format(&surface).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        // A BindGroup describes a set of resources and how they can be accessed by a shader.
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor
        {
            entries: &[
                wgpu::BindGroupLayoutEntry
                {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture
                    {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry
                {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler
                    {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        });

        let camera = Camera
        {
            // Position the camera. +z is out of the screen.
            eye: (0.0, 5.0, -10.0).into(),
            // Have it look at the origin
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: sc_desc.width as f32 / sc_desc.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let camera_controller = CameraController::new(0.2); // Speed is 0.2

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        // We are gonna update camera position, so we need COPY_DST.
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor
        {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        // Space Between Instances.
        const SPACE_BETWEEN: f32 = 4.0;
        // Create instances amount of NUM_INSTANCES_PER_ROW * NUM_INSTANCES_PER_ROW. (x and z axes)
        let instances = (0..NUM_INSTANCES_PER_ROW).flat_map(|z|
        {
            (0..NUM_INSTANCES_PER_ROW).map(move |x|
            {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_ROW as f32 / 2.0);
                let position = cgmath::Vector3 { x, y: 0.0, z };
                let rotation = if position.is_zero()
                {
                    // this is needed so an object at (0, 0, 0) won't get scaled to zero
                    // as Quaternions can effect scale if they're not created correctly.
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(),cgmath::Deg(0.0),)
                } 
                else { cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0)) };
                
                Instance { position, rotation }
            })
        }).collect::<Vec<_>>();
        
        // Create actual instances data.
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        
        // And use it.
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor
        {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsage::VERTEX,
        });

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor
        {
            entries: &[wgpu::BindGroupLayoutEntry
            {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::Buffer
                {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor
        {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry
            {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        // Find the res folder. obj files are in there.
        let res_dir = std::path::Path::new(env!("OUT_DIR")).join("res");
        if res_dir.exists() { println!("\n res folder found: main.rs");}

        // Load obj file. "capsule.obj" or "cube.obj".
        let obj_model = model::Model::load(&device,&queue,&texture_bind_group_layout,res_dir.join("capsule.obj"),)
        .unwrap();
        
        let light_uniform = LightUniform
        {
            position: [1.0, 3.0, 5.0], // Light position
            _padding: 0,
            color: [1.0, 1.0, 1.0], // White light.
        };

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor
        {
            label: Some("Light VB"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        // Create a bind group layout for light.
        let light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor
        {
            entries: &[wgpu::BindGroupLayoutEntry
            {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                ty: wgpu::BindingType::Buffer
                {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: None,
        });

        // create a bind group for light.
        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor
        {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry
            {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });


        let light_render_pipeline =
        {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor
            {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            
            let shader = wgpu::ShaderModuleDescriptor
            {
                label: Some("Light Shader"),
                flags: wgpu::ShaderFlags::all(),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()), // Read light.wgsl
            };
            
            // see the light in the scene
            create_render_pipeline(&device, &layout, sc_desc.format,
                        Some(texture::Texture::DEPTH_FORMAT), &[model::ModelVertex::desc()],shader,)
        };

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor
        {
            label: Some("shader.wgsl"), 
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()), // Read shader.wgsl
            flags: wgpu::ShaderFlags::VALIDATION,
        });

        let depth_texture = texture::Texture::create_depth_texture(&device, &sc_desc, "depth_texture");

        // The PipelineLayout contains a list of BindGroupLayouts that the pipeline can use.
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor
        {
            label: Some("Render Pipeline Layout"),
            // texture - camera - light
            bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout,&light_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor
        {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState
            {
                module: &shader,
                entry_point: "main",
                // The buffers field tells wgpu what type of vertices we want to pass to the vertex shader, .obj and how many.+
                buffers: &[model::ModelVertex::desc(), InstanceRaw::desc()],
            },
            // We need fragment if we want to store color data
            fragment: Some(wgpu::FragmentState
            {
                module: &shader,
                entry_point: "main",
                // The targets field tells wgpu what color outputs it should set up.
                targets: &[wgpu::ColorTargetState
                {
                    format: sc_desc.format,
                    blend: Some(wgpu::BlendState
                    {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrite::ALL,
                }],
            }),
            // The primitive field describes how to interpret our vertices when converting them into triangles.
            primitive: wgpu::PrimitiveState
            {
                // Each three vertices will correspond to one triangle.
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // Determine whether a given triangle is facing forward or not.
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back), 
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLAMPING
                clamp_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState
            {
                format: texture::Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState
            {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: true, // Anti-aliasing
            },
        });
        
        // Return value.
        Self
        {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,
            size,
            render_pipeline,
            obj_model,
            camera,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
            instances,
            instance_buffer,
            depth_texture,
            light_bind_group,
            light_render_pipeline,
        }
    }

    // Need to recreate the swap_chain everytime the window's size changes.
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>)
    {
        if new_size.width > 0 && new_size.height > 0
        {
            self.camera.aspect = self.sc_desc.width as f32 / self.sc_desc.height as f32;
            self.size = new_size;
            self.sc_desc.width = new_size.width;
            self.sc_desc.height = new_size.height;
            self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
            self.depth_texture = texture::Texture::create_depth_texture(&self.device,&self.sc_desc,"depth_texture",);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool
    {
        self.camera_controller.process_events(event)
    }

    fn update(&mut self)
    {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(&self.camera_buffer,0,bytemuck::cast_slice(&[self.camera_uniform]),);
    }
    
    // Here's where the magic happens.
    fn render(&mut self) -> Result<(), wgpu::SwapChainError>
    {
        let frame = self.swap_chain.get_current_frame()?.output;

        // CommandEncoder: create the actual commands to send to the gpu.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor
        {
            label: Some("Render Encoder"),
        });

        // Inner scope
        {   
            // The RenderPass has all the methods to do the actual drawing.
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor
            {
                label: Some("Render Pass"),
                // The RenderPassColorAttachment has the view field which informs wgpu what texture to save the colors to.
                color_attachments: &[wgpu::RenderPassColorAttachment
                {
                    view: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations
                    {
                        // Background color.
                        load: wgpu::LoadOp::Clear(wgpu::Color
                        {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment
                {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations
                    {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            // First parameter is what buffer slot to use for vertices.
            // Second is the slice of the buffer to use. You can store as many objects in a buffer.
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

            use crate::model::DrawLight;
            render_pass.set_pipeline(&self.light_render_pipeline);
            render_pass.draw_light_model(&self.obj_model,&self.camera_bind_group,&self.light_bind_group,); // light
            render_pass.set_pipeline(&self.render_pipeline);
            // render obj as many as intances.len with camera and light.
            render_pass.draw_model_instanced(&self.obj_model,0..self.instances.len() as u32,&self.camera_bind_group,&self.light_bind_group,);


            // for rendering pentagon (deprecated)
            /*
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..self.instances.len() as _); */
        }
        
        // Tell wgpu to finish the command buffer, and to submit it to the gpu's render queue.
        self.queue.submit(iter::once(encoder.finish()));
        
        // Return value.
        Ok(())
    }
}

fn main()
{
    env_logger::init();
    let event_loop = EventLoop::new();
    let title = env!("CARGO_PKG_NAME");
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .build(&event_loop)
        .unwrap();

    // Since main can't be async, we're going to need to block.
    let mut state = pollster::block_on(State::new(&window));

    event_loop.run(move |event, _, control_flow|
    {
        *control_flow = ControlFlow::Poll;
        match event
        {
            Event::MainEventsCleared => window.request_redraw(),
            Event::WindowEvent
            {
                ref event,
                window_id,
            } 
            if window_id == window.id() =>
            {
                if !state.input(event)
                {
                    match event
                    {
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput
                        {
                            input:
                                KeyboardInput
                                {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        } => *control_flow = ControlFlow::Exit,

                        WindowEvent::Resized(physical_size) =>
                        {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } =>
                        {
                            // new_inner_size is &&mut so we have to dereference it twice.
                            state.resize(**new_inner_size);
                        }
                        _ => {} // Exit
                    }
                }
            }
            Event::RedrawRequested(_) =>
            {
                state.update();
                match state.render()
                {
                    Ok(_) => {}
                    // Recreate the swap_chain if lost
                    Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SwapChainError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            _ => {}
        }
    });
}