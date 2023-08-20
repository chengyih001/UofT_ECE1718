#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h> // for SSIM calculation
#include <filesystem>
#include "opencv2/opencv.hpp"

namespace fs = std::filesystem;

std::vector<unsigned char> read_image(const std::string& file_path, int& out_width, int& out_height, int& out_channels, bool do_bicubic, int target_width, int target_height) {
    // Read the image using OpenCV
    cv::Mat img = cv::imread(file_path, cv::IMREAD_UNCHANGED);

    // Check if the image is empty
    if (img.empty()) {
        std::cerr << "ERROR: Unable to read image: " << file_path << std::endl;
        exit(1);
    }
    
    // Convert the image from BGR(A) to RGB(A)
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    // resize the image - bicubic
    if (do_bicubic)
        resize(img, img, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
    
    // Get the image dimensions and channels
    out_width = img.cols;
    out_height = img.rows;
    out_channels = img.channels();

    // Assign the data from the image to the vector and resize it
    std::vector<unsigned char> img_data(img.data, img.data + (out_width * out_height * out_channels));

    return img_data;
}

class SRCNNDataset : public torch::data::Dataset<SRCNNDataset> {
public:
    SRCNNDataset(const std::string& image_paths, const std::string& label_paths) {
        for (const auto& entry : fs::directory_iterator(image_paths)) {
            all_image_paths.push_back(entry.path().string());
        }
        for (const auto& entry : fs::directory_iterator(label_paths)) {
            all_label_paths.push_back(entry.path().string());
        }

        // Sort the paths in alphabetical order (only for validation set)
        if (all_image_paths.size() < 100) {
            std::sort(all_image_paths.begin(), all_image_paths.end());
            std::sort(all_label_paths.begin(), all_label_paths.end());
        }
    }

    torch::data::Example<> get(size_t index) override {
        int image_width, image_height, channels;
        int label_width, label_height;
        std::string image_path = all_image_paths[index]; // for debug only
        std::string label_path = all_label_paths[index]; // for debug only
        std::vector<unsigned char> label_data = read_image(all_label_paths[index], label_width, label_height, channels, false, 0, 0);
        std::vector<unsigned char> image_data = read_image(all_image_paths[index], image_width, image_height, channels, true, label_width, label_height);

        torch::Tensor img_tensor = torch::from_blob(image_data.data(), { image_height, image_width, 3 }, torch::kUInt8).clone();
        torch::Tensor label_tensor = torch::from_blob(label_data.data(), { label_height, label_width, 3 }, torch::kUInt8).clone();

        // Normalize image data to range [0, 1]
        img_tensor = img_tensor.to(torch::kFloat32) / 255.0;
        label_tensor = label_tensor.to(torch::kFloat32) / 255.0;

        img_tensor = img_tensor.permute({ 2, 0, 1 });
        label_tensor = label_tensor.permute({ 2, 0, 1 });

        return { img_tensor, label_tensor };
    }

    torch::optional<size_t> size() const override {
        return all_image_paths.size();
    }

private:
    std::vector<std::string> all_image_paths;
    std::vector<std::string> all_label_paths;
};

float ssim_single_channel(torch::Tensor source, torch::Tensor target, float K1 = 0.01, float K2 = 0.03) {
    source = source.unsqueeze(0);
    target = target.unsqueeze(0);

    // Ensure the input tensors have the same shape and are 1-channel images (one of the RGB channel)
    assert(source.sizes() == target.sizes());
    assert(source.dim() == 3 && source.size(0) == 1);

    source = source.unsqueeze(0);
    target = target.unsqueeze(0);

    int C1 = (K1 * 255) * (K1 * 255);
    int C2 = (K2 * 255) * (K2 * 255);
    int window_size = 11;
    int padding = (window_size - 1) / 2;

    // Create a Gaussian kernel for windowing - to perform Gaussian bluer
    torch::Tensor window = torch::hann_window(window_size).unsqueeze(1) * torch::hann_window(window_size).unsqueeze(0);
    window /= window.sum().item<float>();
    window = window.expand({ source.size(1), 1, window_size, window_size });

    // Convert the images to the device of the input tensors
    window = window.to(source.device(), source.dtype());

    // Compute means
    torch::Tensor mu1 = torch::conv2d(source, window, {}, 1, padding).squeeze(0);
    torch::Tensor mu2 = torch::conv2d(target, window, {}, 1, padding).squeeze(0);

    // Compute the mean of the products
    torch::Tensor mu1_mu2 = mu1 * mu2;

    // Compute the squares of the means
    mu1 = mu1 * mu1;
    mu2 = mu2 * mu2;

    // Compute the variances
    torch::Tensor sigma1 = torch::conv2d(source * source, window, {}, 1, padding) - mu1;
    torch::Tensor sigma2 = torch::conv2d(target * target, window, {}, 1, padding) - mu2;

    // Compute the covariance
    torch::Tensor sigma12 = torch::conv2d(source * target, window, {}, 1, padding) - mu1_mu2;

    // Compute the SSIM map
    torch::Tensor ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 + mu2 + C1) * (sigma1 + sigma2 + C2));

    // Compute the mean SSIM
    float mean_ssim = ssim_map.mean().item<float>();

    return mean_ssim;
}

float ssim(torch::Tensor source, torch::Tensor target, float K1 = 0.01, float K2 = 0.03) {
    // do SSIM for each channel first, then take the average to get the SSIM contributs all 3 channels
    double SSIM_R = ssim_single_channel(source[0][0], target[0][0], K1, K2);
    double SSIM_G = ssim_single_channel(source[0][1], target[0][1], K1, K2);
    double SSIM_B = ssim_single_channel(source[0][2], target[0][2], K1, K2);
    double SSIM = (SSIM_R + SSIM_G + SSIM_B) / 3;
    return SSIM;
}

double psnr(torch::Tensor source, torch::Tensor target, double max_value) {
    torch::Tensor mse = torch::mse_loss(source, target);
    double mse_value = mse.item<double>();
    if (mse_value == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return 20 * std::log10(max_value / std::sqrt(mse_value));
}

int const TEST_BATCH_SIZE = 1;

int main() {
    // torch::manual_seed(0);

    // set image paths
    std::string valid_image_dir = "/homes/s/shuteng/Desktop/ECE1718/asgmnt3/milestone2/SRCNN_python/input/test_bicubic_rgb_2x";
    std::string valid_label_dir = "/homes/s/shuteng/Desktop/ECE1718/asgmnt3/milestone2/SRCNN_python/input/test_hr";
    
    // Create the SRCNNDataset
    auto valid_dataset = SRCNNDataset(valid_image_dir, valid_label_dir).map(torch::data::transforms::Stack<>());
    auto validation_set_size = valid_dataset.size();

    // create data_loaders
    auto valid_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(valid_dataset), TEST_BATCH_SIZE);

    // Get the start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // validation loop
    size_t batch_idx = 0;
    double valid_total_PSNR = 0;
    double valid_final_PSNR = 0;
    double valid_total_SSIM = 0;
    double valid_final_SSIM = 0;

    // Iterate through the DataLoader for validation
    for (auto& batch : *valid_data_loader) {
        // Get the images and labels, and move them to the device
        torch::Tensor images = batch.data;
        torch::Tensor labels = batch.target;

        // calculate the PSNR
        double PSNR = psnr(images, labels, (double)1);
        valid_total_PSNR += PSNR;

        // calculate the SSIM
        double SSIM = ssim(images, labels);
        valid_total_SSIM += SSIM;
        std::cout << "PSNR = " << PSNR << ", SSIM = " << SSIM << "\n";
        batch_idx++;
    }
    valid_final_PSNR = valid_total_PSNR / batch_idx;
    valid_final_SSIM = valid_total_SSIM / batch_idx;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    auto duration_minutes = static_cast<double>(duration_seconds) / 60.0;
    //print some information
    std::cout << "Done validation, average PSNR(per batch) = " << valid_final_PSNR << ", average SSIM(per batch) = " << valid_final_SSIM << "\n";
    std::cout << "Finished inferencing (libTorch) in: " << duration_minutes << " minutes" << std::endl;
    return 0;
}
