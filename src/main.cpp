#include <algorithm>
#include <utility>
#include <vector>
#include <filesystem>
#include <iostream>
#include <sys/stat.h>
#include <iostream>
#include <boost/filesystem.hpp>

#include "config_parser.hpp"

std::string
replaceOtherStr(std::string &replacedStr, std::string from, std::string to) {
  const unsigned int pos = replacedStr.find(from);
  const int len = from.length();

  if (pos == std::string::npos || from.empty()) {
    return replacedStr;
  }

  return replacedStr.replace(pos, len, to);
}

void
saveImage(cv::Mat &img, const std::string &dir, const std::string &name)
{
  fs::path p = dir;
  p.append(name);
  std::string dst = p.string();
  std::cout << "##Save " << dst << std::endl;
  cv::imwrite(dst, img);
}

int
main(int argc, char* argv[])
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::string model_path = get_onnx_path();
  std::string precision = get_precision();
  std::string directory = get_directory_path();
  std::string video_path = get_video_path();
  int cam_id = get_camera_id();    
  std::string output_image_path = "dst.png";
  std::string calibration_images = get_calibration_images();
  const int batch = get_batch_size();
  const bool dont_show = is_dont_show();
  const int dla = get_dla_id();
  const bool first = get_fisrt_flg();
  const bool last = get_last_flg();
  std::string calibType = get_calib_type();
  bool prof = get_prof_flg();
  double clip = get_clip_value();
  float max_depth = get_max_depth();
  Window_info window_info = get_window_info();
  
  const tensorrt_common::BatchConfig & batch_config = {1, batch/2, batch};
  const size_t workspace_size = (1 << 30);
  tensorrt_common::BuildConfig build_config(
					    calibType, dla, first, last,
					    prof, clip);
  auto trt_depth_anything = std::make_unique<tensorrt_depth_anything::TrtDepth_Anything>(model_path, precision, build_config, false, calibration_images, batch_config, workspace_size);

  bool init = true;
  bool flg_save = getSaveDetections();
  std::string save_path = getSaveDetectionsPath();

  if (flg_save) {
    fs::create_directory(save_path);
  }
  
  if (directory != "") {
    std::vector<std::string> filenames;
    for (const auto & file : std::filesystem::directory_iterator(directory)) {
      filenames.push_back(file.path());
    }    
    for (int i = 0; i < (int)filenames.size(); i++) {
      std::vector<cv::Mat> inputs;
      std::cout << "Infer from ... " << std::endl;
      std::cout << filenames[i] << std::endl;
      auto image = cv::imread(filenames[i], cv::IMREAD_UNCHANGED);	  
      if (init) {
	trt_depth_anything->initPreprocessBuffer(image.cols, image.rows);
	init = false;
      }
      inputs.push_back(image);

      trt_depth_anything->doInference(inputs);

      std::string cFormat = get_depth_colormap();  
      cv::Mat depth = trt_depth_anything->getDepthImage(cFormat, max_depth);
      const auto scale_size = cv::Size(image.cols, image.rows);
      cv::resize(depth, depth, scale_size, 0, 0, cv::INTER_NEAREST);
      cv::namedWindow("depth", cv::WINDOW_NORMAL);
      cv::imshow("depth", depth);      

      for (int b = 0; b < batch; b++) {
	cv::Mat src = inputs[b];
       
	if (!dont_show) {
	  cv::namedWindow("src image" + std::to_string(b), cv::WINDOW_NORMAL);
	  cv::imshow("src image" + std::to_string(b), src);
	}	
	if (flg_save) {
	  fs::path p (filenames[i+b]);
	  std::string name = p.filename().string();	
	  std::ostringstream sout;
	  p = save_path;
	  replaceOtherStr(name, ".jpg", ".png");	  
	  saveImage(depth, p.string(), name);
	}
	if (!dont_show) {
	  cv::waitKey(0);
	}
      }
    }
  } else if (video_path != "" || cam_id != -1) {
   std::cout << video_path << std::endl;
    cv::VideoCapture video;
    if (cam_id != -1) {
      video.open(cam_id);
    } else {
      video.open(video_path);
    }
    cv::Mat image;
    std::string window_name = "inference image";
    cv::namedWindow(window_name, 0);
    if (window_info.w !=0 && window_info.h !=0) {
      cv::resizeWindow(window_name, window_info.w, window_info.h);
    }
    cv::moveWindow(window_name, window_info.x, window_info.y);
    int frame_count = 0;
    while (1) {
      std::vector<cv::Mat> inputs;
      video >> image;
      if (image.empty() == true) break;
      if (init) {
	trt_depth_anything->initPreprocessBuffer(image.cols, image.rows);
	init = false;
      }
      inputs.push_back(image);

      trt_depth_anything->doInference(inputs);

      std::string cFormat = get_depth_colormap();  
      cv::Mat depth = trt_depth_anything->getDepthImage(cFormat, max_depth);
      const auto scale_size = cv::Size(image.cols, image.rows);
      cv::resize(depth, depth, scale_size, 0, 0, cv::INTER_NEAREST);
      cv::namedWindow("depth", cv::WINDOW_NORMAL);
      cv::imshow("depth", depth);      

      cv::Mat src = inputs[0];       
      if (!dont_show) {
	  cv::namedWindow("src image" + std::to_string(0), cv::WINDOW_NORMAL);
	  cv::imshow("src image" + std::to_string(0), src);
      }	
      if (flg_save) {

      }
      if (cv::waitKey(1) == 'q') break;
      frame_count++;
    } 
  }    
  
  if (prof) {
    trt_depth_anything->printProfiling();
  }

  return 0;
}
