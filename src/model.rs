use anyhow::*;
use std::ops::Range;
use std::path::Path;
use tobj::LoadOptions;
use wgpu::util::DeviceExt;

use crate::texture;

pub trait Vertex
{   
    // Like attributes in OpneGL.
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex
{
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
}

impl Vertex for ModelVertex
{   
    // Like attributes in OpneGL.
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>
    {
        use std::mem;
        wgpu::VertexBufferLayout
        {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                // Positions
                wgpu::VertexAttribute
                {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Texcoords
                wgpu::VertexAttribute
                {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // Normals
                wgpu::VertexAttribute
                {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub struct Material
{
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

// Mesh holds a vertex buffer, an index buffer, and the number of indices in the mesh.
pub struct Mesh
{
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,    // for Vertices
    pub index_buffer: wgpu::Buffer,     // for Indices
    pub num_elements: u32,
    pub material: usize,
}

// obj file can include multiple meshes and materials.
pub struct Model
{
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

impl Model
{
    // .obj file read:
    pub fn load<P: AsRef<Path>>(device: &wgpu::Device,queue: &wgpu::Queue,layout: &wgpu::BindGroupLayout,path: P,) -> Result<Self>
    {
        let (obj_models, obj_materials) = tobj::load_obj(path.as_ref(),&LoadOptions
        {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },)?;

        let obj_materials = obj_materials?;

        // We're assuming that the texture files are stored with the obj file.
        let containing_folder = path.as_ref().parent().context("Directory has no parent")?;

        let mut materials = Vec::new();
        for mat in obj_materials
        {
            let diffuse_path = mat.diffuse_texture;
            let diffuse_texture = texture::Texture::load(device, queue, containing_folder.join(diffuse_path))?;
            println!(" diffuse_texture loaded: model.rs");

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor
            {
                layout,
                entries: &[
                    wgpu::BindGroupEntry
                    {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry
                    {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    },
                ],
                label: None,
            });

            materials.push(Material
            {
                name: mat.name,
                diffuse_texture,
                bind_group,
            });
        }
        println!(" materials filled: model.rs");

        let mut meshes = Vec::new();
        for m in obj_models
        {
            // Something familier :
            let mut vertices = Vec::new();
            for i in 0..m.mesh.positions.len() / 3
            {
                vertices.push(ModelVertex {
                    // We got position datas from mesh to our ModelVertex Struct.
                    position: [ 
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    // We got texcoord datas from mesh to our ModelVertex Struct.
                    tex_coords: [
                        m.mesh.texcoords[i * 2],
                        m.mesh.texcoords[i * 2 + 1]
                    ],
                    // We got normal datas from mesh to our ModelVertex Struct.
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                });
            }
            println!(" obj_model vertices assigned: model.rs");

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", path.as_ref())),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsage::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", path.as_ref())),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsage::INDEX,
            });

            meshes.push(Mesh
            {
                name: m.name,
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            });
            println!(" meshes assigned: model.rs");
        }
        println!(" obj_model loaded succesfully: model.rs\n");

        // Return Value.
        Ok(Self { meshes, materials })
    }
}

pub trait DrawModel<'a>
{
    // We invoke this func in the main.rs. This func. invokes draw_mesh_instanced as many as the range in a for loop.
    fn draw_model_instanced(&mut self,model: &'a Model,instances: Range<u32>,camera: &'a wgpu::BindGroup,light: &'a wgpu::BindGroup,);
    
    // Buffers and bind groups for model.
    fn draw_mesh_instanced(&mut self,mesh: &'a Mesh,material: &'a Material,instances: Range<u32>,camera: &'a wgpu::BindGroup,light: &'a wgpu::BindGroup,);
}

impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,  // if b's type is a ??
{    
    fn draw_model_instanced(&mut self,model: &'b Model,instances: Range<u32>,camera: &'b wgpu::BindGroup,light: &'a wgpu::BindGroup,)
    {
        for mesh in &model.meshes
        {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone(), camera,light);
        }
    }

    fn draw_mesh_instanced(&mut self,mesh: &'b Mesh,material: &'b Material,instances: Range<u32>,camera: &'b wgpu::BindGroup,light: &'a wgpu::BindGroup,)
    {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, &material.bind_group, &[]);
        self.set_bind_group(1, camera, &[]);
        self.set_bind_group(2, light, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }
}

pub trait DrawLight<'a>
{   
    // We invokde this func in main.rs.  This func invkoes draw_light_model_intanced.
    fn draw_light_model(&mut self,model: &'a Model,camera: &'a wgpu::BindGroup,light: &'a wgpu::BindGroup,);

    // This func invokes draw_light_mesh_instanced as many as the range value in a for loop.
    fn draw_light_model_instanced(&mut self,model: &'a Model,instances: Range<u32>,camera: &'a wgpu::BindGroup,light: &'a wgpu::BindGroup,);
    
    // Buffers and bind groups for light.
    fn draw_light_mesh_instanced(&mut self,mesh: &'a Mesh,instances: Range<u32>,camera: &'a wgpu::BindGroup,light: &'a wgpu::BindGroup,); 
}

impl<'a, 'b> DrawLight<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,  // if b's type is a ??
{
    fn draw_light_model(&mut self,model: &'b Model,camera: &'b wgpu::BindGroup,light: &'b wgpu::BindGroup,)
    {
        self.draw_light_model_instanced(model, 0..1, camera, light);
    }

    fn draw_light_model_instanced(&mut self,model: &'b Model,instances: Range<u32>,camera: &'b wgpu::BindGroup,light: &'b wgpu::BindGroup,)
    {
        for mesh in &model.meshes
        {
            self.draw_light_mesh_instanced(mesh, instances.clone(), camera, light);
        }
    }

    fn draw_light_mesh_instanced(&mut self,mesh: &'b Mesh,instances: Range<u32>,camera: &'b wgpu::BindGroup,light: &'b wgpu::BindGroup,)
    {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera, &[]);
        self.set_bind_group(1, light, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }
}