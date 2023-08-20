#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
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

        // Normalize image data to range 0-1
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

struct CustomNetImpl : torch::nn::Module {
    CustomNetImpl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 9).stride(1).padding(2)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 1).stride(1).padding(2)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 3, 5).stride(1).padding(2)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = conv3->forward(x);

        return x;
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
};

TORCH_MODULE(CustomNet);

torch::Tensor convolution_forward_pass(int kernel_size,
    int in_channels,
    int out_channels,
    bool apply_relu,
    int input_height,
    int input_width,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {
    int pad = 2;
    int stride = 1;

    // Calculate the output dimensions
    int output_height = (input_height + 2 * pad - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * pad - kernel_size) / stride + 1;

    // Create the output tensor
    torch::Tensor output = torch::zeros({ out_channels, output_height, output_width });

    // initialize data accessors
    auto input_squeezed = input.squeeze(0);
    auto input_accessor = input_squeezed.accessor<float, 3>();
    auto weights_accessor = weights.accessor<float, 4>();
    auto bias_accessor = bias.accessor<float, 1>();
    auto output_accessor = output.accessor<float, 3>();

    for (int oc = 0; oc < out_channels; oc++) {
        for (int h = 0; h < output_height; h++) {
            for (int w = 0; w < output_width; w++) {
                float sum = 0;
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int h_in = h + kh - pad;
                            int w_in = w + kw - pad;

                            if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                                sum += input_accessor[ic][h_in][w_in] * weights_accessor[oc][ic][kh][kw];
                            }
                        }
                    }
                }
                sum += bias_accessor[oc];
                if (apply_relu && sum < 0) {
                    sum = 0;
                }
                output_accessor[oc][h][w] = sum;
            }
        }
    }
    return output;
}

torch::Tensor manual_forward_pass(torch::Tensor input,
    torch::Tensor weights1, torch::Tensor bias1,
    torch::Tensor weights2, torch::Tensor bias2,
    torch::Tensor weights3, torch::Tensor bias3) {
    
    // Define some parameters
    int in_channels = 3;
    int conv1_kernel_size = 9;
    int conv1_out_channels = 64;
    int conv2_kernel_size = 1;
    int conv2_out_channels = 32;
    int conv3_kernel_size = 5;
    int conv3_out_channels = 3;

    torch::Tensor conv1_output = convolution_forward_pass(conv1_kernel_size, in_channels, conv1_out_channels, true, input.sizes()[2], input.sizes()[3], input, weights1, bias1);
    torch::Tensor conv2_output = convolution_forward_pass(conv2_kernel_size, conv1_out_channels, conv2_out_channels, true, conv1_output.sizes()[1], conv1_output.sizes()[2], conv1_output, weights2, bias2);
    torch::Tensor conv3_output = convolution_forward_pass(conv3_kernel_size, conv2_out_channels, conv3_out_channels, false, conv2_output.sizes()[1], conv2_output.sizes()[2], conv2_output, weights3, bias3);

    return conv3_output;
}

double psnr(torch::Tensor source, torch::Tensor target, double max_value) {
    torch::Tensor mse = torch::mse_loss(source, target);
    double mse_value = mse.item<double>();
    return 20 * std::log10(max_value / std::sqrt(mse_value));
}

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

int const TRAIN_BATCH_SIZE = 128;
int const TEST_BATCH_SIZE = 1;

int main() {
    torch::manual_seed(0);

    // set image paths
    std::string valid_image_dir = "./test_bicubic_rgb_2x";
    std::string valid_label_dir = "./test_hr";
    
    // Create the SRCNNDataset
    auto valid_dataset = SRCNNDataset(valid_image_dir, valid_label_dir).map(torch::data::transforms::Stack<>());
    auto validation_set_size = valid_dataset.size();

    // create data_loaders
    auto valid_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(valid_dataset), TEST_BATCH_SIZE);

    // load the network
    CustomNet network;
    std::string model_path = "./traced_resnet_model.pt";
    torch::load(network, model_path);
    std::cout << network << "\n";

    // this implementation only uses CPU
    torch::Device device = torch::Device(torch::kCPU);
    network->to(device);

    // read its weights and bias
    torch::Tensor conv1_weights = network->conv1->weight;
    torch::Tensor conv1_bias = network->conv1->bias;
    torch::Tensor conv2_weights = network->conv2->weight;
    torch::Tensor conv2_bias = network->conv2->bias;
    torch::Tensor conv3_weights = network->conv3->weight;
    torch::Tensor conv3_bias = network->conv3->bias;

    // start validation using manual forward pass
    size_t batch_idx = 0;
    double valid_total_PSNR = 0;
    double valid_final_PSNR = 0;
    double valid_total_SSIM = 0;
    double valid_final_SSIM = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (auto& batch : *valid_data_loader) {
        //batch_idx = 1;
        //break;

        // Get the images and labels, and move them to the device
        torch::Tensor images = batch.data.to(device);
        torch::Tensor labels = batch.target.to(device);

        torch::Tensor output = manual_forward_pass(images, conv1_weights, conv1_bias, conv2_weights, conv2_bias, conv3_weights, conv3_bias).unsqueeze(0);

        // calculate the PSNR
        double PSNR = psnr(output, labels, (double)1);
        valid_total_PSNR += PSNR;

        // calculate the SSIM
        double SSIM = ssim(output, labels);
        valid_total_SSIM += SSIM;
        std::cout << "PSNR = " << PSNR << ", SSIM = " << SSIM << "\n";

        batch_idx++;
    }
    valid_final_PSNR = valid_total_PSNR / batch_idx;
    valid_final_SSIM = valid_total_SSIM / batch_idx;
    auto end_time = std::chrono::high_resolution_clock::now();
	auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double duration_minutes = static_cast<double>(duration_ms) / 1000.0 / 60.0;
    //print some information
    std::cout << "Done validation, average PSNR(per batch) = " << valid_final_PSNR << ", average SSIM(per batch) = " << valid_final_SSIM << "\n";
    std::cout << "Finished inferencing (manual forward pass) in: " << duration_ms/1000.0 << " seconds" << std::endl;
    std::cout << "                                     which is: " << duration_minutes << " minutes" << std::endl;

    // start validation using libTorch
    network->eval();
    batch_idx = 0;
    valid_total_PSNR = 0;
    valid_final_PSNR = 0;
    valid_total_SSIM = 0;
    valid_final_SSIM = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (auto& batch : *valid_data_loader) {
        torch::NoGradGuard no_grad;

        // Get the images and labels, and move them to the device
        torch::Tensor images = batch.data.to(device);
        torch::Tensor labels = batch.target.to(device);

        torch::Tensor output = network->forward(images);

        // calculate the PSNR
        double PSNR = psnr(output, labels, (double)1);
        valid_total_PSNR += PSNR;

        // calculate the SSIM
        double SSIM = ssim(output, labels);
        valid_total_SSIM += SSIM;
        std::cout << "PSNR = " << PSNR << ", SSIM = " << SSIM << "\n";

        batch_idx++;
    }
    valid_final_PSNR = valid_total_PSNR / batch_idx;
    valid_final_SSIM = valid_total_SSIM / batch_idx;
    end_time = std::chrono::high_resolution_clock::now();
	duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    duration_minutes = static_cast<double>(duration_ms) / 1000.0 / 60.0;
    //print some information
    std::cout << "Done validation, average PSNR(per batch) = " << valid_final_PSNR << ", average SSIM(per batch) = " << valid_final_SSIM << "\n";
    std::cout << "Finished inferencing (libTorch) in: " << duration_ms/1000.0 << " seconds" << std::endl;
    std::cout << "                          which is: " << duration_minutes << " minutes" << std::endl;
}

