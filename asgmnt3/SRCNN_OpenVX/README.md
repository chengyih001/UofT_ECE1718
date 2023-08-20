# Milestone 3 Environment Setup

## Setup LibTorch
go to the directory that you will save all these libraries
```
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
rm libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
```
(Mac using the url below, not tested)
```
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.0.zip
unzip libtorch-macos-2.0.0.zip
rm libtorch-macos-2.0.0.zip
```
take note of the path of the ```.../libtorch/```

## Setup OpenCV
go to the directory that you will save all these libraries
```
mkdir -p opencv && cd opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
rm opencv.zip
mkdir -p build && cd build
cmake  ../opencv-4.x
cmake --build .
export OpenCV_DIR=$(pwd)
```
take note of the path of the ```.../opencv/build/```

## Setup OpenVX (without AMD implementation)
go to the directory that you will save all these libraries
```
git clone --recursive https://github.com/KhronosGroup/OpenVX-sample-impl.git
cd OpenVX-sample-impl/
python Build.py --os=Linux --arch=64 --conf=Debug --conf_vision --enh_vision --conf_nnef
export OPENVX_DIR=$(pwd)/install/Linux/x64/Debug
```
take note of the path of the ```.../OpenVX-sample-impl/install/Linux/x64/Debug/```

## Get an Example Code and Test It
go to the directory that you will save your code
```
git clone https://github.com/kiritigowda/openvx-samples.git
mkdir build && cd build
cmake -DOPENVX_INCLUDES=$OPENVX_DIR/include -DOPENVX_LIBRARIES=$OPENVX_DIR/bin/libopenvx.so ../openvx-samples/canny-edge-detector/
make
```

## Modify the Example Code
```rm``` the unnecessory exmaple code directories and files except ```CMakeLists.txt``` and ```canny-edge-detector/```\
```rm``` the code ```canny-edge-detector/src/canny.cpp``` and replace with your cpp file ```<your file name>.cpp```\
change the directory name of ```canny-edge-detector/``` to ```<project name>/```
### Modify ```CMakeLists.txt``` under ```openvx-samples/```
change the last three lines of ```add_subdirectory(...)``` to one line of ```add_subdirectory(<project name>)```

### Modify ```CMakeLists.txt``` under ```openvx-samples/<project name>/```
change the line ```project(cannyEdgeDetector)``` to ```project(<project name>)```\
add the following code under the line ```project(<project name>)```
```
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH <your LibTorch path>)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")
``` 
change the line ```add_executable(${PROJECT_NAME} src/canny.cpp)``` to ```add_executable(${PROJECT_NAME} src/<your file name>.cpp)```\
change the line ```target_link_libraries(${PROJECT_NAME} ${OPENVX_LIBRARIES} ${OpenCV_LIBRARIES})``` to ```target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OPENVX_LIBRARIES} ${OpenCV_LIBRARIES})```

## Redo cmake and Run Your Code
go to the directory that you cloned the example project
```
cd build
rm * -rf
cmake -DOPENVX_INCLUDES=$OPENVX_DIR/include -DOPENVX_LIBRARIES=$OPENVX_DIR/bin/libopenvx.so ../openvx-samples/<project name>/
make
./<your file name>
```
