#include <memory>
#include <string>
#include <gflags/gflags.h>
#include <map>
#include <NvInfer.h>
#include <tensorrt_depth_anything/tensorrt_depth_anything.hpp>

typedef struct _window_info{
  unsigned int x;
  unsigned int y;
  unsigned int w;
  unsigned int h;
} Window_info;

std::string
get_onnx_path(void);

std::string
get_directory_path(void);

std::string
get_video_path(void);

int
get_camera_id(void);
  
std::string
get_precision(void);

bool
is_dont_show(void);

std::string
get_calibration_images();

bool
get_prof_flg(void);
int
get_batch_size(void);
int
get_width(void);

int
get_height(void);

int
get_dla_id(void);

bool
get_fisrt_flg(void);

bool
get_last_flg(void);

std::string
get_dump_path(void);

std::string
get_calib_type(void);

double
get_clip_value(void);

bool getSaveDetections();
std::string getSaveDetectionsPath();

Window_info
get_window_info(void);

std::string
get_depth_colormap(void);

float
get_max_depth(void);
