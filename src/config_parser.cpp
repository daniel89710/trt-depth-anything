#include "config_parser.hpp"
#include <assert.h>
#include <iostream>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>

DEFINE_string(onnx, "yolox-tiny.onnx",
              "ONNX Path, "
              "ONNX Path");

DEFINE_bool(dont_show, false,
	    "[Optional] Flag to off screen");

DEFINE_string(d, "",
              "Directory Path, "
              "Directory Path");

DEFINE_string(v, "",
              "Video Path, "
              "Video Path");

DEFINE_int64(cam_id, -1, "Camera ID");

DEFINE_string(calibration_table_path, "not-specified",
              "[OPTIONAL] Path to pre-generated calibration table. If flag is not set, a new calib "
              "table <network-type>-<precision>-calibration.table will be generated");

DEFINE_string(precision, "FP32",
              "[OPTIONAL] Inference precision. Choose from kFLOAT, kHALF and kINT8.");
DEFINE_string(deviceType, "kGPU",
              "[OPTIONAL] The device that this layer/network will execute on. Choose from kGPU and kDLA(only for kHALF).");


DEFINE_string(calibration_images, "calibration_images.txt",
             "[OPTIONAL] Text file containing absolute paths or filenames of calibration images. "
              "Flag required if precision is kINT8 and there is not pre-generated calibration "
              "table. If only filenames are provided, their corresponding source directory has to "
              "be provided through 'calibration_images_path' flag");


DEFINE_string(depth, "gray",
              "[OPTIONAL] Depth color format (gray/magma/jet)");

DEFINE_double(max_depth, 0.0, "[OPTIONAL] thresh");

DEFINE_uint64(batch_size, 1, "[OPTIONAL] Batch size for the inference engine.");
DEFINE_uint64(width, 0, "[OPTIONAL] width for the inference engine.");
DEFINE_uint64(height, 0, "[OPTIONAL] height for the inference engine.");
DEFINE_int64(dla, -1, "[OPTIONAL] DLA");


DEFINE_bool(prof, false,
            "[OPTIONAL] Flag to profile layer by layer");

DEFINE_bool(first, false,
            "[OPTIONAL] Flag to keep high precision for first layer");

DEFINE_bool(last, false,
            "[OPTIONAL] Flag to keep high precision for last layer");

DEFINE_string(calib, "MinMax",
              "[OPTIONAL] Calibration Type");

DEFINE_double(clip, 0.0, "[OPTIONAL] Clip value for implicit quantization in output");

DEFINE_bool(save_detections, false,
            "[OPTIONAL] Flag to save images overlayed with objects detected.");
DEFINE_string(save_detections_path, "outputs/",
              "[OPTIONAL] Path where the images overlayed with bounding boxes are to be saved");

DEFINE_uint64(wx, 0, "[OPTIONAL] position x for display window");
DEFINE_uint64(wy, 0, "[OPTIONAL] position y for display window");
DEFINE_uint64(ww, 0, "[OPTIONAL] width for display window");
DEFINE_uint64(wh, 0, "[OPTIONAL] height for display window");

std::string
get_onnx_path(void)
{
  return FLAGS_onnx;
}

std::string
get_directory_path(void)
{
  return FLAGS_d;
}

int
get_camera_id(void)
{
  return FLAGS_cam_id;
}

std::string
get_video_path(void)
{
  return FLAGS_v;
}

std::string
get_precision(void)
{
  return FLAGS_precision;
}

bool
is_dont_show(void)
{
  return FLAGS_dont_show;
}

std::string
get_calibration_images()
{
  return FLAGS_calibration_images;
}

bool
get_prof_flg(void)
{
  return FLAGS_prof;
}

int
get_batch_size(void)
{
  return FLAGS_batch_size;
}

int
get_width(void)
{
  return FLAGS_width;
}

int
get_height(void)
{
  return FLAGS_height;
}

int
get_dla_id(void)
{
  return FLAGS_dla;
}

bool
get_fisrt_flg(void)
{
  return FLAGS_first;
}

bool
get_last_flg(void)
{
  return FLAGS_last;
}

std::string
get_calib_type(void)
{
  std::string type;
  if (FLAGS_calib == "Entropy") {
    type =  FLAGS_calib;
  } else if (FLAGS_calib == "Legacy" || FLAGS_calib == "Percentile") {
    type = FLAGS_calib;
  } else {
    type = "MinMax";
  }
  return type;
}


double
get_clip_value(void)
{
  return FLAGS_clip;
}

static bool isFlagDefault(std::string flag) { return flag == "not-specified" ? true : false; }

bool getSaveDetections()
{
  if (FLAGS_save_detections)
    assert(!isFlagDefault(FLAGS_save_detections_path)
	   && "save_detections path has to be set if save_detections is set to true");
  return FLAGS_save_detections;
}

std::string getSaveDetectionsPath() { return FLAGS_save_detections_path; }

Window_info
get_window_info(void)
{
  Window_info winfo = {(unsigned int)FLAGS_wx, (unsigned int)FLAGS_wy, (unsigned int)FLAGS_ww, (unsigned int)FLAGS_wh};
  return winfo;
}

std::string
get_depth_colormap(void)
{
  return FLAGS_depth;
}

float
get_max_depth(void)
{
  return (float)FLAGS_max_depth;
}
