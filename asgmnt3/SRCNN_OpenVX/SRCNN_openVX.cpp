#include <iostream>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <VX/vx_khr_import_kernel.h>
#include "opencv2/opencv.hpp"
#include <torch/torch.h>
#include <filesystem>

#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
}

#define ERROR_CHECK_OBJECT( obj ) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
}

namespace fs = std::filesystem;

// this is fixed due to the limitation of the ONNX and NNEF models
const int width = 512;
const int height = 512;
const size_t TEST_BATCH_SIZE = 1;

std::vector<unsigned char> read_image(const std::string& file_path, int& out_width, int& out_height, int& out_channels) {
    // Read the image using OpenCV
    cv::Mat img = cv::imread(file_path, cv::IMREAD_UNCHANGED);

    // Check if the image is empty
    if (img.empty()) {
        std::cerr << "ERROR: Unable to read image: " << file_path << std::endl;
        exit(1);
    }
    
    // Convert the image from BGR(A) to RGB(A)
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    // resize the image - due to NNEF model only support fixed size input and output
    resize(img, img, cv::Size(height, width), 0, 0, cv::INTER_CUBIC);
    
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
        
        std::vector<unsigned char> image_data = read_image(all_image_paths[index], image_width, image_height, channels);
        std::vector<unsigned char> label_data = read_image(all_label_paths[index], label_width, label_height, channels);

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

struct ConvLayerImpl : torch::nn::Module {
    ConvLayerImpl(int in_channels, int out_channels, int kernel_size)
        : conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
            .padding(2)) {
        register_module("conv", conv);
    }

    torch::Tensor forward(torch::Tensor x, bool do_relu) {
        auto weight = conv->weight;
        auto bias = conv->bias;
        int height = x.sizes()[2];
        int width = x.sizes()[3];
        int kernel_size = conv->options.kernel_size()->at(0);
        int padding = 2;

        x = torch::conv2d(x, weight, bias, 1, padding);
        
        if (do_relu == true) {
            x = torch::relu(x);
        }
        return x;
    }

    torch::nn::Conv2d conv{ nullptr };
};

TORCH_MODULE(ConvLayer);

struct CustomNetImpl : torch::nn::Module {
    CustomNetImpl()
        : conv1(ConvLayer(3, 64, 9)),
        conv2(ConvLayer(64, 32, 1)),
        conv3(ConvLayer(32, 3, 5)) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv1(x, true);
        x = conv2(x, true);
        x = conv3(x, false); // no Relu in this layer
        return x;
    }

    ConvLayer conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr };
};

TORCH_MODULE(CustomNet);

void cp_TorchTensor_To_OpenVXTensor(const torch::Tensor &torch_tensor, vx_tensor openvx_tensor, vx_context context) {
    // Check if the input tensor is a 4D tensor
    if (torch_tensor.dim() != 4) {
        printf("ERROR: Input tensor must be a 4D tensor!");
        exit(1);
    }

    // Get the dimensions of the input tensor
    auto sizes = torch_tensor.sizes();
    vx_size dims[4] = {static_cast<vx_size>(sizes[0]), static_cast<vx_size>(sizes[1]), static_cast<vx_size>(sizes[2]), static_cast<vx_size>(sizes[3])};

    // assume the two tensor are having the same size, copy the input tensor data to the OpenVX tensor
    torch::Tensor flat_tensor = torch_tensor.flatten();
    vx_size view_start[4] = {0, 0, 0, 0};
    vx_size view_end[4] = {dims[0], dims[1], dims[2], dims[3]};
    vx_size inputTensorStride[4] = {0, 0, 0, 0};
    inputTensorStride[0] = sizeof(vx_float32);
    for (int j = 1; j < 4; j++)
    {
        inputTensorStride[j] = inputTensorStride[j - 1] * dims[j - 1];
    }
    
    vx_status status = vxCopyTensorPatch(openvx_tensor, 4, view_start, view_end, inputTensorStride, flat_tensor.data_ptr(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    ERROR_CHECK_STATUS(status);
    
    return;
}

torch::Tensor OpenVXTensor_To_TorchTensor(vx_tensor openvx_tensor) {
    // Get the dimensions of the OpenVX tensor
    vx_size dims[4];
    ERROR_CHECK_STATUS(vxQueryTensor(openvx_tensor, VX_TENSOR_DIMS, &dims, sizeof(dims[0]) * 4));

    // Allocate memory for the tensor data
    std::vector<float> tensor_data(dims[0] * dims[1] * dims[2] * dims[3]);

    // Copy the OpenVX tensor data to the allocated memory
    vx_size view_start[4] = {0, 0, 0, 0};
    vx_size view_end[4] = {dims[0], dims[1], dims[2], dims[3]};
    vx_size inputTensorStride[4] = {0, 0, 0, 0};
    inputTensorStride[0] = sizeof(vx_float32);
    for (int j = 1; j < 4; j++) {
        inputTensorStride[j] = inputTensorStride[j - 1] * dims[j - 1];
    }

    vx_status status = vxCopyTensorPatch(openvx_tensor, 4, view_start, view_end, inputTensorStride, tensor_data.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    ERROR_CHECK_STATUS(status);

    // Create a torch::Tensor from the copied data and reshape it to match the dimensions of the OpenVX tensor
    torch::Tensor torch_tensor = torch::from_blob(tensor_data.data(), {static_cast<int64_t>(dims[0]), static_cast<int64_t>(dims[1]), static_cast<int64_t>(dims[2]), static_cast<int64_t>(dims[3])}, torch::kFloat32).clone();

    return torch_tensor;
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

template <class DataLoader>
void validation_libTorch(DataLoader &valid_data_loader) {
    // load the network
    CustomNet network;
    std::string model_path = "/root/autodl-tmp/ECE1718_assignment3/milestone2/model.pt";
    torch::load(network, model_path);
    std::cout << network << "\n";

    // this implementation only uses CPU
    torch::Device device = torch::Device(torch::kCPU);
    network->to(device);
    
    network->eval();
    size_t batch_idx = 0;
    double valid_total_PSNR = 0;
    double valid_final_PSNR = 0;
    double valid_total_SSIM = 0;
    double valid_final_SSIM = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
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
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    auto duration_minutes = static_cast<double>(duration_seconds) / 60.0;
    //print some information
    std::cout << "Done validation, average PSNR(per batch) = " << valid_final_PSNR << ", average SSIM(per batch) = " << valid_final_SSIM << "\n";
    std::cout << "Finished inferencing (libTorch) in: " << duration_minutes << " minutes" << std::endl;
}

int main() {
    // set image paths
    std::string valid_image_dir = "test_bicubic_rgb_2x";
    std::string valid_label_dir = "test_hr";
    
    // Create the SRCNNDataset
    auto valid_dataset = SRCNNDataset(valid_image_dir, valid_label_dir).map(torch::data::transforms::Stack<>());
    auto validation_set_size = valid_dataset.size();
    
    // create data_loaders
    auto valid_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(valid_dataset), TEST_BATCH_SIZE);

    // create OpenVX context
    vx_status status;
    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    
    // create OpenVX Graph
    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);

    //Get nnef kernel
    const char* model_url = "new_model.nnef";
    char nn_type[5] = "nnef";
    vx_char *nnef_type = nn_type;
    vx_kernel mNN_kernel = vxImportKernelFromURL(context, nnef_type, model_url);
    ERROR_CHECK_OBJECT(mNN_kernel);
    printf("vxImportKernelFromURL() for %s model: %s successful\n", nnef_type, model_url);
    
    // query number of parameters in imported kernel
    //vx_int32 num_params = 0;
    //ERROR_CHECK_STATUS(vxQueryKernel(mNN_kernel, VX_KERNEL_PARAMETERS, &num_params, sizeof(vx_uint32)));
    
    // create input and output tensor
    vx_size dims[4] = {1, 3, height, width};
    vx_enum data_type = VX_TYPE_FLOAT32;
    vx_tensor input_tensor  = vxCreateTensor(context, 4, dims, data_type, 0);
    vx_tensor output_tensor = vxCreateTensor(context, 4, dims, data_type, 0);
    ERROR_CHECK_OBJECT(input_tensor);
    ERROR_CHECK_OBJECT(output_tensor);
        
    // create nn node for the graph
    vx_node node = vxCreateGenericNode(graph, mNN_kernel);
    ERROR_CHECK_OBJECT(node);
    printf("vxCreateGenericNode for Imported Kernel successful \n");

    // FIXME: need to use GPU somehow
    //ERROR_CHECK_STATUS(vxSetNodeTarget(node, VX_TARGET_ANY, NULL));

    //add input and output tensors to the node
    status = vxSetParameterByIndex(node, 0, (vx_reference)input_tensor);
    ERROR_CHECK_STATUS(status);
    status = vxSetParameterByIndex(node, 1, (vx_reference)output_tensor);
    ERROR_CHECK_STATUS(status);

    //verify the graph
    status = vxVerifyGraph(graph);
    ERROR_CHECK_STATUS(status);
    
    // just for testing
    //validation_libTorch(valid_data_loader);

    // do inferencing using openVX
    size_t batch_idx = 0;
    double valid_total_PSNR = 0;
    double valid_final_PSNR = 0;
    double valid_total_SSIM = 0;
    double valid_final_SSIM = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    printf("start validation\n");
    for (auto& batch : *valid_data_loader) {
        // Get the images and labels, and move them to the device
        torch::Tensor images = batch.data;
        torch::Tensor labels = batch.target;
        
        // load the input_tensor with data
        cp_TorchTensor_To_OpenVXTensor(images, input_tensor, context);

        //process the graph
        status = vxProcessGraph(graph);
        ERROR_CHECK_STATUS(status);
        printf("graph successfully processed\n");
        
        // extract the output_tensor value to a torch::Tensor
        torch::Tensor output = OpenVXTensor_To_TorchTensor(output_tensor);
        std::cout << "output size = " << output.sizes() << "\n";
        
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
    auto duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    auto duration_minutes = static_cast<double>(duration_seconds) / 60.0;
    //print some information
    std::cout << "Done validation, average PSNR(per batch) = " << valid_final_PSNR << ", average SSIM(per batch) = " << valid_final_SSIM << "\n";
    std::cout << "Finished inferencing (OpenVX) in: " << duration_minutes << " minutes" << std::endl;
    
    return 0;
}
