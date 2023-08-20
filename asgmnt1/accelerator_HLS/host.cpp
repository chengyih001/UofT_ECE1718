#include "host.h"

unsigned char label_test[10000];
unsigned char data_test[10000][784];

size_t get_file_size(const char* filename)
{
    struct stat buf;
    if(stat(filename, &buf)<0)
    {
        return 0;
    }
    return (size_t)buf.st_size;
}

void load_data_to_vec(std::vector<d_type, aligned_allocator<d_type>> &vec, const char* filename) {
    int data_num = get_file_size(filename)/sizeof(d_type);
	d_type* data_buf = (d_type *)calloc(data_num, sizeof(d_type));
	FILE *fp_w = fopen(filename, "rb");
	fread(data_buf, sizeof(d_type), data_num, fp_w);
	fclose(fp_w);
    for (int i=0; i < data_num; i++) {
        vec[i] = data_buf[i];
    }
    free(data_buf);
}

d_type* load_data(const char* filename)
{
    int data_num = get_file_size(filename)/4;
	d_type* data_buf = (d_type *)calloc(data_num, sizeof(d_type));
	FILE *fp_w = fopen(filename, "rb");
	fread(data_buf, sizeof(d_type), data_num, fp_w);
	fclose(fp_w);
    return data_buf;
}

void give_img(unsigned char* vec , d_type img[][32]) {
	int k=0;
	for (int i=0; i<35; i++) {
		for (int j=0; j<32; j++) {
			if (i<5 || j<2 || i>32 || j>29) {
				img[i][j] = 0;
			} else {
				img[i][j] = vec[k++];
			}
		}
	}
}

void read_test_data() {
	ifstream csvread;
	csvread.open("data/mnist_test.csv", ios::in);
	if(csvread) {
		string s;
		int data_pt = 0;
		while(getline(csvread, s)) {
			stringstream ss(s);
			int pxl = 0;
			while( ss.good() ) {
				string substr;
				getline(ss, substr,',');
				if (pxl == 0) {
					label_test[data_pt] = stoi(substr);
				} else {
					data_test[data_pt][pxl-1] = stoi(substr);
				}
				pxl++;
			}
			data_pt++;
		}
		csvread.close();
	}
	else{
		cerr << "Unable to read test data!" << endl;
	exit (EXIT_FAILURE);
	}
}

int give_prediction(d_type arr[CLASS_SIZE]) {
	d_type max_val = arr[0];
	int max_pos = 0;
	for (int i=1; i<10; i++) {
		if (arr[i] > max_val) {
			max_val = arr[i];
			max_pos = i;
		}
	}
	return max_pos;
}

int main(int argc, char** argv) {
	if (argc != 2) {
		std::cout << "Usage " << argv[0] << " <XCLBIN FILE>" << std:endl;
		return EXIT_FAILURE;
	}



 	std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_vadd;
    cl::CommandQueue queue;

	read_test_data();

	d_type output[N_TEST][CLASS_SIZE];

	for (int i=0; i < N_TEST; i++) {
		for (int j=0; j < CLASS_SIZE; j++) {
			output[i][j] = 0;
		}
	}

	std::vector<std::vector<d_type, aligned_allocator<d_type>>> in_test(N_TEST, std::vector<d_type, aligned_allocator<d_type>>(36*32, 0));

	std::vector<d_type, aligned_allocator<d_type>> in(36*32);
	std::vector<d_type, aligned_allocator<d_type>> conv_w(FILTER_SIZE*CONV_SIZE*CONV_SIZE);
	std::vector<d_type, aligned_allocator<d_type>> conv_b(FILTER_SIZE*28*28);
	std::vector<d_type, aligned_allocator<d_type>> dense1_w(DENSE_1*DENSE_2);
	std::vector<d_type, aligned_allocator<d_type>> dense1_b(DENSE_2);
	std::vector<d_type, aligned_allocator<d_type>> dense2_w(DENSE_2*CLASS_SIZE);
	std::vector<d_type, aligned_allocator<d_type>> dense2_b(CLASS_SIZE);
	std::vector<d_type, aligned_allocator<d_type>> out(CLASS_SIZE, 0);

	const int MAX_OUT = 3920;
	std::vector<d_type, aligned_allocator<d_type>> temp_out(MAX_OUT, 0);

	load_data_to_vec(conv_w, "parameters/conv_w.bin");
	load_data_to_vec(conv_b, "parameters/conv_b.bin");
	load_data_to_vec(dense1_w, "parameters/dense1_w.bin");
	load_data_to_vec(dense1_b, "parameters/dense1_b.bin");
	load_data_to_vec(dense2_w, "parameters/dense2_w.bin");
	load_data_to_vec(dense2_b, "parameters/dense2_b.bin");


	int cor=0;
	int confusion_mat[10][10];
	for (int i=0; i<10; i++){
		for (int j=0; j<10; j++) confusion_mat[i][j] = 0;
	}

	
	// OpenCL Host Code
    auto xilinx_devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i=0; i < xilinx_devices.size(); i++) {
        OCL_CHECK(err, context = cl::Context(xilinx_devices[i], nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, queue = cl::CommandQueue(context, xilinx_devices[i], CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Programming xilinx_device " << i << " :" << xilinx_devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {xilinx_devices[i]}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed while programming xilinx_device " << i << " with xclbin file." << std::endl;
        }
        else {
            std::cout << "Xilinx_device " << i << " programmed successfully." << std::endl;
            OCL_CHECK(err, krnl_vadd = cl::Kernel(program, "vadd", &err));
            valid_device = true;
            break;
        }
    }
    if (!valid_device) {
        std::cout << "No device available!" << std::endl;
        exit(EXIT_FAILURE);
    }

	// Allocate device memory
	OCL_CHECK(err, cl::Buffer buffer_conv_w(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, FILTER_SIZE*CONV_SIZE*CONV_SIZE*sizeof(d_type), conv_w.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_conv_b(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, FILTER_SIZE*28*28*sizeof(d_type), conv_b.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_dense1_w(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, DENSE_1*DENSE_2*sizeof(d_type), dense1_w.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_dense1_b(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, DENSE_2*sizeof(d_type), dense1_b.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_dense2_w(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, DENSE_2*CLASS_SIZE*sizeof(d_type), dense2_w.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_dense2_b(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, CLASS_SIZE*sizeof(d_type), dense2_b.data(), &err));
	OCL_CHECK(err, cl::Buffer buffer_out(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, CLASS_SIZE*sizeof(d_type), out.data(), &err));

	OCL_CHECK(err, cl::Buffer buffer_temp_out(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, MAX_OUT*sizeof(d_type), temp_out.data(), &err));


	// Copy data to device memory
	OCL_CHECK(err, err = queue.enqueueMigrateMemObjects{buffer_conv_w, buffer_conv_b,
														buffer_dense1_w, buffer_dense1_b,
														buffer_dense2_w, buffer_dense2_b,
														temp_out, out}, 0);
	
	
	for (int i=0; i < N_TEST; i++) {
		d_type img[35][32];
		give_img(data_test[i], img);
		for (int j=0; j < 36; j++) {
			for (int k=0; k < 32; k++) {
				in_test[i][j*32+k] = img[j][k];
			}
		}
	}

	cout << "Start Testing." << endl;

	for (int i=0; i < N_TEST; i++) {
		for (int j=0; j < 36*32; j++) {
			in[j] = in_test[i][j];
		}

		// Set kernel args
		OCL_CHECK(err, err = krnl_vadd.setArg(0, buffer_in));
		OCL_CHECK(err, err = krnl_vadd.setArg(1, buffer_out));
		OCL_CHECK(err, err = krnl_vadd.setArg(2, buffer_conv_w));
		OCL_CHECK(err, err = krnl_vadd.setArg(3, buffer_conv_b));
		OCL_CHECK(err, err = krnl_vadd.setArg(4, buffer_dense1_w));
		OCL_CHECK(err, err = krnl_vadd.setArg(5, buffer_dense1_b));
		OCL_CHECK(err, err = krnl_vadd.setArg(6, buffer_dense2_w));
		OCL_CHECK(err, err = krnl_vadd.setArg(7, buffer_dense2_b));
		OCL_CHECK(err, err = krnl_vadd.setArg(7, buffer_temp_out));


		
		// Copy input to device memory
		OCL_CHECK(err, cl::Buffer buffer_in(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 36*32*sizeof(d_type), in_test[i].data(), &err));
		OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buffer_in}, 0));

		// Launch kernel
		OCL_CHECK(err, err = queue.enqueueTask(krnl_vadd));

		// Copy result back to host
		OCL_CHECK(err, err = queue.enqueueMigrateMemObjects({buffer_out}, CL_MIGRATE_MEM_OBJECT_HOST));
		OCL_CHECK(err, err = queue.finish());

		for (int j=0; j < CLASS_SIZE; j++) {
			output[i][j] = out[j];
		}
	}

	for (int i=0; i < N_TEST; i++) {
		int pre = give_prediction(output[i]);
		confusion_mat[label_test[i]][pre]++;
		if (pre == label_test[i]) cor++;
	}

	float accu = float(cor)/val_len;
	cout << "Accuracy: " << accu << endl;

	cout << "   0 1 2 3 4 5 6 7 8 9" << endl;
	for (int i=0; i<10; i++){
		cout << i << ": ";
		for (int j=0; j<10; j++) {
			cout << confusion_mat[i][j] << " ";
		}
		cout << endl;
	}

	return 0;

	

}