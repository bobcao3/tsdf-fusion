# --------------------------------------------------------
# Cheng Cao, Berkeley 2019
# --------------------------------------------------------
# Translated and extended from C
# Original Author: Andy Zeng, Princeton University, 2016
# --------------------------------------------------------

using CUDAdrv, CUDAnative, CuArrays
using DelimitedFiles, Printf, FileIO, Images
using MeshCat
using CoordinateTransformations
using GeometryTypes: GeometryTypes, Vec, Point, Point3f0
using Colors: RGBA, RGB

@enum occupancy::UInt8 begin
    unknown = 0
    occupied = 1
    free = 255
end

function Integrate(
    cam_K, cam2base, cam2world,
    depth_im, im_height, im_width,
    voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z, 
    voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
    voxel_size, trunc_margin,
    voxel_grid_TSDF, voxel_grid_weight, voxel_grid_occupancy
)
    pt_grid_z = blockIdx().x
    pt_grid_y = threadIdx().x

    for pt_grid_x::Int = 1:voxel_grid_dim_x
        pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size
        pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size
        pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size

        tmp_pt_1 = pt_base_x - cam2base[1, 4]
        tmp_pt_2 = pt_base_x - cam2base[2, 4]
        tmp_pt_3 = pt_base_x - cam2base[3, 4]

        pt_cam_x = cam2base[1, 1] * tmp_pt_1 + cam2base[2, 1] * tmp_pt_2 + cam2base[3, 1] * tmp_pt_3
        pt_cam_y = cam2base[1, 2] * tmp_pt_1 + cam2base[2, 2] * tmp_pt_2 + cam2base[3, 2] * tmp_pt_3
        pt_cam_z = cam2base[1, 3] * tmp_pt_1 + cam2base[2, 3] * tmp_pt_2 + cam2base[3, 3] * tmp_pt_3

        if pt_cam_z <= 0
            continue
        end

        pt_pix_x::Int = round(cam_K[1, 1] * (pt_cam_x / pt_cam_z) + cam_K[1, 3])
        pt_pix_y::Int = round(cam_K[2, 2] * (pt_cam_x / pt_cam_z) + cam_K[2, 3])
        if pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height
            continue
        end

        depth_val = depth_im[pt_pix_y, pt_pix_x]

        diff = depth_val - pt_cam_z

        if diff > 0 && voxel_grid_occupancy[pt_grid_x, pt_grid_y, pt_grid_z] != occupied
            voxel_grid_occupancy[pt_grid_x, pt_grid_y, pt_grid_z] = free
        end

        if depth_val <= 0 || depth_val > 6
            continue
        end

        if diff <= -trunc_margin
            continue
        end

        dist = min(1.0, diff / trunc_margin)
        weight_old = voxel_grid_weight[pt_grid_x, pt_grid_y, pt_grid_z]
        weight_new = weight_old + 1.0
        voxel_grid_weight[pt_grid_x, pt_grid_y, pt_grid_z] = weight_new
        tsdf = (voxel_grid_TSDF[pt_grid_x, pt_grid_y, pt_grid_z] * weight_old + dist) / weight_new
        voxel_grid_TSDF[pt_grid_x, pt_grid_y, pt_grid_z] = tsdf

        if abs(tsdf) < voxel_size / trunc_margin && weight_new > 0.0
            voxel_grid_occupancy[pt_grid_x, pt_grid_y, pt_grid_z] = occupied
        end
    end

    return
end    

function main(args)
    # Location of camera intrinsic file
    cam_K_file = "data/camera-intrinsics.txt";

    # Location of folder containing RGB-D frames and camera pose files
    data_path = "data/rgbd-frames";
    base_frame_idx::Int = 150;
    first_frame_idx::Int = 150;
    num_frames::Int = 50;

    cam_K = Array{Float32, 2}(undef, 3, 3)
    base2world = Array{Float32, 2}(undef, 4, 4)
    cam2base = Array{Float32, 2}(undef, 4, 4)
    cam2world = Array{Float32, 2}(undef, 4, 4)
    cam2world_inv = Array{Float32, 2}(undef, 4, 4)
    im_width::Int = 640;
    im_height::Int = 480;
    depth_im = Array{Float32, 2}(undef, im_width, im_height);

    # Voxel grid parameters (change these to change voxel grid resolution, etc.)
    voxel_grid_origin = Float32[-1.5 -1.5 0.5] # Location of voxel grid origin in base frame camera coordinates
    voxel_size::Float32 = 0.006
    trunc_margin::Float32 = voxel_size * 5.0
    voxel_grid_dim = [500 500 500]

    # Manual parameters
    if length(args) > 1
        cam_K = readdlm(args[1], ' ', Float32)
        data_path = args[2];
        base_frame_idx = parse(Int, args[3]);
        first_frame_idx = parse(Int, args[4]);
        num_frames = parse(Int, args[5]);
        voxel_grid_origin = [parse(Float32, args[6]) parse(Float32, args[7]) parse(Float32, args[8])]
        voxel_size = parse(Float32, args[9]);
        trunc_margin = parse(Float32, args[10]);
    end

    @show cam_K data_path base_frame_idx first_frame_idx num_frames voxel_grid_origin voxel_size trunc_margin

    base2world = Float32[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    base2world_inv = inv(base2world)

    voxel_grid_TSDF = fill(1.0, (voxel_grid_dim[1], voxel_grid_dim[2], voxel_grid_dim[3]))
    voxel_grid_weight = fill(0.0, (voxel_grid_dim[1], voxel_grid_dim[2], voxel_grid_dim[3]))
    voxel_grid_occupancy = fill(unknown, (voxel_grid_dim[1], voxel_grid_dim[2], voxel_grid_dim[3]))

    gpu_voxel_grid_TSDF = CuArray(voxel_grid_TSDF)
    gpu_voxel_grid_weight = CuArray(voxel_grid_weight)
    gpu_voxel_grid_occupancy = CuArray(voxel_grid_occupancy)

    gpu_cam_K = CuArray(cam_K)
    gpu_cam2base = CuArray(cam2base)
    gpu_cam2world = CuArray(cam2world)
    gpu_voxel_grid_dim = CuArray(voxel_grid_dim)
    gpu_voxel_grid_origin = CuArray(voxel_grid_origin)
    
    for frame_idx::Int = first_frame_idx:first_frame_idx + num_frames
        depth_im_file = @sprintf "%s/frame-%06d.depth.png" data_path frame_idx
        depth_im = convert(Array{Float32, 2}, Gray.(load(depth_im_file)))

        cam2world_file = @sprintf "%s/frame-%06d.pose.txt" data_path frame_idx
        cam2world = readdlm(cam2world_file, ' ', Float32)

        base2world_inv = cam2world * cam2base;

        cam2world_inv = inv(cam2world)

        gpu_cam2base = CuArray(cam2base)
        gpu_cam2world = CuArray(cam2world)
        gpu_depth_im = CuArray(depth_im)

        println("Fusing: ", depth_im_file)

        @cuda blocks=voxel_grid_dim[3] threads=voxel_grid_dim[2] Integrate(
            gpu_cam_K, gpu_cam2base, gpu_cam2world,
            gpu_depth_im, im_height, im_width,
            gpu_voxel_grid_dim[1], gpu_voxel_grid_dim[2], gpu_voxel_grid_dim[3],
            gpu_voxel_grid_origin[1], gpu_voxel_grid_origin[2], gpu_voxel_grid_origin[3],
            voxel_size, trunc_margin,
            gpu_voxel_grid_TSDF, gpu_voxel_grid_weight, gpu_voxel_grid_occupancy
        )
    end

    voxel_grid_TSDF = Array(gpu_voxel_grid_TSDF)
    voxel_grid_occupancy = Array(gpu_voxel_grid_occupancy)

    write("tsdf.bin", voxel_grid_occupancy)

    vis = Visualizer()
    open(vis)

    verts = Point3f0[]
    colors = RGBA[]

    for x = 1:voxel_grid_dim[1]
        for y = 1:voxel_grid_dim[2]
            for z = 1:voxel_grid_dim[3]
                #if voxel_grid_occupancy[x, y, z] == occupied
                    push!(verts, Point3f0(
                        x * voxel_size + voxel_grid_origin[1],
                        y * voxel_size + voxel_grid_origin[2],
                        z * voxel_size + voxel_grid_origin[3]
                    ))
                    push!(colors, RGBA(1.0, 1.0, 1.0, voxel_grid_TSDF[x, y, z]))
                #end
            end
        end
        println(x)
    end

    setobject!(vis, PointCloud(verts, colors))

    return readline()
end

main(ARGS)