// ---------------------------------------------------------
// Author: Andy Zeng, Princeton University, 2016
// ---------------------------------------------------------

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "utils.hpp"
#include "helper_math.h"

__device__
float3 matrix_mul(float* K, float3 v) {
  float tmp_pt[3] = {0};
  tmp_pt[0] = v.x - K[0 * 4 + 3];
  tmp_pt[1] = v.y - K[1 * 4 + 3];
  tmp_pt[2] = v.z - K[2 * 4 + 3];
  float x = K[0 * 4 + 0] * tmp_pt[0] + K[1 * 4 + 0] * tmp_pt[1] + K[2 * 4 + 0] * tmp_pt[2];
  float y = K[0 * 4 + 1] * tmp_pt[0] + K[1 * 4 + 1] * tmp_pt[1] + K[2 * 4 + 1] * tmp_pt[2];
  float z = K[0 * 4 + 2] * tmp_pt[0] + K[1 * 4 + 2] * tmp_pt[1] + K[2 * 4 + 2] * tmp_pt[2];
  return make_float3(x, y, z);
}

__device__
bool test_segment(float i, int r0, int r1) {
  return i * 255 > r0 - 0.002 && i * 255 < r1 + 0.002;
}

// CUDA kernel function to integrate a TSDF voxel volume given depth images
__global__
void Integrate(float * cam_K, float * instance_K, float * cam2base, float * cam2world, cudaTextureObject_t depth_tex, cudaTextureObject_t instance_tex,
               int instance_im_height, int instance_im_width,
               int im_height, int im_width, int3 voxel_grid_dim,
               float3 voxel_grid_origin, float voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight, char * voxel_grid_occupancy) {

  int3 pt_grid = make_int3(0, blockIdx.x, threadIdx.x);

  for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim.x; ++pt_grid_x) {
    pt_grid.x = (float) pt_grid_x;

    float3 pt_base = voxel_grid_origin + make_float3(pt_grid) * voxel_size;

    // Convert from base frame camera coordinates to current frame camera coordinates
    float3 pt_cam = matrix_mul(cam2base, pt_base);

    if (pt_cam.z <= 0)
      continue;

    float pt_pix_x = (cam_K[0 * 3 + 0] * (pt_cam.x / pt_cam.z) + cam_K[0 * 3 + 2]);
    float pt_pix_y = (cam_K[1 * 3 + 1] * (pt_cam.y / pt_cam.z) + cam_K[1 * 3 + 2]);
    if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
      continue;

    float pt_ins_x = (instance_K[0 * 3 + 0] * (pt_cam.x / pt_cam.z) + instance_K[0 * 3 + 2]);
    float pt_ins_y = (instance_K[1 * 3 + 1] * (pt_cam.y / pt_cam.z) + instance_K[1 * 3 + 2]);

    float depth_val = tex2D<float>(depth_tex, pt_pix_x / (float) im_width, pt_pix_y / (float) im_height);
    float segment = tex2D<float>(instance_tex, pt_ins_x / (float) instance_im_width, pt_ins_y / (float) instance_im_height);

    float diff = depth_val - pt_cam.z;

    int volume_idx = pt_grid.z * voxel_grid_dim.y * voxel_grid_dim.x + pt_grid.y * voxel_grid_dim.x + pt_grid.x;

    if (diff > 0)
      voxel_grid_occupancy[volume_idx] = FREE;

    if (depth_val <= 0 || depth_val > 6)
    continue;

    if (diff <= -trunc_margin)
      continue;

    // Integrate
    float dist = fmin(1.0f, diff / trunc_margin);
    float weight_old = voxel_grid_weight[volume_idx];
    float weight_new = weight_old + 1.0f;
    voxel_grid_weight[volume_idx] = weight_new;
    voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;

    if (std::abs(voxel_grid_TSDF[volume_idx]) < voxel_size / trunc_margin * (0.5 * 1.414) && weight_new > 0.0 && (
      test_segment(segment, 0, 0) ||
      test_segment(segment, 15, 16) ||
      test_segment(segment, 20, 22)
    )) {
      voxel_grid_occupancy[volume_idx] = OCCUPIED;
    }
  }
}

// Loads a binary file with depth data and generates a TSDF voxel volume (5m x 5m x 5m at 1cm resolution)
// Volume is aligned with respect to the camera coordinates of the first frame (a.k.a. base frame)
int main(int argc, char * argv[]) {

  // Location of camera intrinsic file
  std::string cam_K_file = "data/camera-intrinsics.txt";
  std::string instance_K_file = "data/instance-intrinsics.txt";

  // Location of folder containing RGB-D frames and camera pose files
  std::string data_path = "data/rgbd-frames";
  int base_frame_idx = 150;
  int first_frame_idx = 150;
  float num_frames = 50;

  float cam_K[3 * 3];
  float instance_K[3 * 3];
  float base2world[4 * 4];
  float cam2base[4 * 4];
  float cam2world[4 * 4];
  float cam2world_inv[4 * 4];
  int im_width = 640;
  int im_height = 480;
  float depth_im[im_height * im_width];
  int instance_im_width = 1296;
  int instance_im_height = 968;
  float instance_im[instance_im_height * instance_im_width];

  // Voxel grid parameters (change these to change voxel grid resolution, etc.)
  float voxel_grid_origin_x = -1.5f; // Location of voxel grid origin in base frame camera coordinates
  float voxel_grid_origin_y = -1.5f;
  float voxel_grid_origin_z = 0.5f;
  float voxel_size = 0.006f;
  float trunc_margin = voxel_size * 5;
  int voxel_grid_dim_x = 500;
  int voxel_grid_dim_y = 500;
  int voxel_grid_dim_z = 500;

  // Manual parameters
  if (argc > 1) {
    cam_K_file = argv[1];
    data_path = argv[2];
    base_frame_idx = atoi(argv[3]);
    first_frame_idx = atoi(argv[4]);
    num_frames = atof(argv[5]);
    voxel_grid_origin_x = atof(argv[6]);
    voxel_grid_origin_y = atof(argv[7]);
    voxel_grid_origin_z = atof(argv[8]);
    voxel_size = atof(argv[9]);
    trunc_margin = atof(argv[10]);
    instance_K_file = argv[11];
  }

  std::cout << cam_K_file << " " << data_path << " " << base_frame_idx << " " << first_frame_idx << " " << num_frames << std::endl;

  // Read camera intrinsics
  std::vector<float> cam_K_vec = LoadMatrixFromFile(cam_K_file, 3, 3);
  std::copy(cam_K_vec.begin(), cam_K_vec.end(), cam_K);

  std::vector<float> instance_K_vec = LoadMatrixFromFile(instance_K_file, 3, 3);
  std::copy(instance_K_vec.begin(), instance_K_vec.end(), instance_K);

  std::cout << "Camera Intrinsics read" << std::endl;

  // Read base frame camera pose
  std::ostringstream base_frame_prefix;
  base_frame_prefix << std::setw(6) << std::setfill('0') << base_frame_idx;
  std::string base2world_file = data_path + "/frame-" + base_frame_prefix.str() + ".pose.txt";
  std::vector<float> base2world_vec = {
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0};//LoadMatrixFromFile(base2world_file, 4, 4);
  std::copy(base2world_vec.begin(), base2world_vec.end(), base2world);

  std::cout << "Base frame pose read" << std::endl;

  // Invert base frame camera pose to get world-to-base frame transform 
  float base2world_inv[16] = {0};
  invert_matrix(base2world, base2world_inv);

  // Initialize voxel grid
  float * voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  float * voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  char * voxel_grid_occupancy = new char[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
    voxel_grid_TSDF[i] = 1.0f;
  memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);
  memset(voxel_grid_occupancy, UNKNOWN, sizeof(char) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);

  // Load variables to GPU memory
  float * gpu_voxel_grid_TSDF;
  float * gpu_voxel_grid_weight;
  char * gpu_voxel_grid_occupancy;
  cudaMalloc(&gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
  cudaMalloc(&gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
  cudaMalloc(&gpu_voxel_grid_occupancy, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(char));
  checkCUDA(__LINE__, cudaGetLastError());
  cudaMemcpy(gpu_voxel_grid_TSDF, voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_voxel_grid_weight, voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_voxel_grid_occupancy, voxel_grid_occupancy, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(char), cudaMemcpyHostToDevice);
  checkCUDA(__LINE__, cudaGetLastError());
  float * gpu_cam_K;
  float * gpu_instance_K;
  float * gpu_cam2base;
  float * gpu_cam2world;
  cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float));
  cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&gpu_instance_K, 3 * 3 * sizeof(float));
  cudaMemcpy(gpu_instance_K, instance_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&gpu_cam2base, 4 * 4 * sizeof(float));
  cudaMalloc(&gpu_cam2world, 4 * 4 * sizeof(float));
  checkCUDA(__LINE__, cudaGetLastError());

  // Allocate array and copy image data
  cudaChannelFormatDesc channelDesc =
  cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray *cuArray;
  cudaMallocArray(&cuArray, &channelDesc, im_width, im_height);

  cudaTextureObject_t         tex;
  cudaResourceDesc            texRes;
  memset(&texRes,0,sizeof(cudaResourceDesc));

  texRes.resType            = cudaResourceTypeArray;
  texRes.res.array.array    = cuArray;

  cudaTextureDesc             texDescr;
  memset(&texDescr,0,sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = true;
  texDescr.filterMode       = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.addressMode[1] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL);

  cudaChannelFormatDesc instanceChannelDesc =
  cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray *cuArrayInstance;
  cudaMallocArray(&cuArrayInstance, &instanceChannelDesc, instance_im_width, instance_im_height);

  cudaTextureObject_t         insTex;
  cudaResourceDesc            insTexRes;
  memset(&insTexRes,0,sizeof(cudaResourceDesc));

  insTexRes.resType            = cudaResourceTypeArray;
  insTexRes.res.array.array    = cuArrayInstance;

  cudaTextureDesc             instanceTexDescr;
  memset(&instanceTexDescr,0,sizeof(cudaTextureDesc));

  instanceTexDescr.normalizedCoords = true;
  instanceTexDescr.filterMode       = cudaFilterModeLinear;
  instanceTexDescr.addressMode[0] = cudaAddressModeClamp;
  instanceTexDescr.addressMode[1] = cudaAddressModeClamp;
  instanceTexDescr.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(&insTex, &insTexRes, &instanceTexDescr, NULL);

  // Loop through each depth frame and integrate TSDF voxel grid
  for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; ++frame_idx) {

    std::ostringstream curr_frame_prefix;
    curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;

    // // Read current frame depth
    std::string depth_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".depth.png";
    ReadDepth(depth_im_file, im_height, im_width, depth_im);

    std::string instance_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".instance.png";
    ReadDepth(instance_im_file, instance_im_height, instance_im_width, instance_im);

    // Read base frame camera pose
    std::string cam2world_file = data_path + "/frame-" + curr_frame_prefix.str() + ".pose.txt";
    std::vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
    std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

    // Compute relative camera pose (camera-to-base frame)
    multiply_matrix(base2world_inv, cam2world, cam2base);

    invert_matrix(cam2world, cam2world_inv);

    cudaMemcpy(gpu_cam2base, cam2base, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_cam2world, cam2world_inv, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());

    cudaMemcpy2DToArray(cuArray, 0, 0, depth_im, im_width * sizeof(float), im_width * sizeof(float), im_height, cudaMemcpyHostToDevice);
    cudaMemcpy2DToArray(cuArrayInstance, 0, 0, instance_im, instance_im_width * sizeof(float), instance_im_width * sizeof(float), instance_im_height, cudaMemcpyHostToDevice);

    std::cout << "Fusing: " << depth_im_file << std::endl;

    Integrate <<< voxel_grid_dim_z, voxel_grid_dim_y >>> (gpu_cam_K, gpu_instance_K, gpu_cam2base, gpu_cam2world, tex, insTex, instance_im_height, instance_im_width,
                                                          im_height, im_width, make_int3(voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z),
                                                          make_float3(voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z), voxel_size, trunc_margin,
                                                          gpu_voxel_grid_TSDF, gpu_voxel_grid_weight, gpu_voxel_grid_occupancy);
  }

  // Load TSDF voxel grid from GPU to CPU memory
  cudaMemcpy(voxel_grid_TSDF, gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(voxel_grid_weight, gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(voxel_grid_occupancy, gpu_voxel_grid_occupancy, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(bool), cudaMemcpyDeviceToHost);
  checkCUDA(__LINE__, cudaGetLastError());

  // Compute surface points from TSDF voxel grid and save to point cloud .ply file
  std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
  SaveVoxelGrid2SurfacePointCloud("tsdf.ply", voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z, 
                                  voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                  voxel_grid_TSDF, voxel_grid_weight, voxel_grid_occupancy, 0.2f, 0.0f);

  {
    // Save TSDF voxel grid and its parameters to disk as binary file (float array)
    std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
    std::string voxel_grid_saveto_path = "tsdf.bin";
    std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
    float voxel_grid_dim_xf = (float) voxel_grid_dim_x;
    float voxel_grid_dim_yf = (float) voxel_grid_dim_y;
    float voxel_grid_dim_zf = (float) voxel_grid_dim_z;
    outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
    outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
    outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
    outFile.write((char*)&voxel_grid_origin_x, sizeof(float));
    outFile.write((char*)&voxel_grid_origin_y, sizeof(float));
    outFile.write((char*)&voxel_grid_origin_z, sizeof(float));
    outFile.write((char*)&voxel_size, sizeof(float));
    outFile.write((char*)&trunc_margin, sizeof(float));
    for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
      outFile.write((char*)&voxel_grid_TSDF[i], sizeof(float));
    outFile.close();
  }

  // Save voxel occupancy grid and its parameters to disk as binary file (float array)
  {
    std::cout << "Saving voxel occupancy grid values to disk (occupancy.bin)..." << std::endl;
    std::string voxel_grid_saveto_path = "occupancy.bin";
    std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
    for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
      outFile.write((char*)&voxel_grid_occupancy[i], sizeof(char));
    outFile.close();
  }

  return 0;
}


