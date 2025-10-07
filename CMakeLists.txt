cmake_minimum_required(VERSION 3.18)

project(DIP_Lab LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add option to enable/disable Qt usage
option(ENABLE_QT "Enable Qt Widgets for file dialog" ON)
 
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(OpenCV REQUIRED)

if(ENABLE_QT)
    find_package(Qt6 COMPONENTS Widgets REQUIRED)
    message(STATUS "Qt support: ENABLED")
else()
    message(STATUS "Qt support: DISABLED")
endif()

find_package(CUDAToolkit REQUIRED)

file(GLOB CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")

set(WITH_OPENCV ${CPP_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/mains/main.cpp)
set(WITH_CUDA ${CU_SOURCES} ${CPP_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/mains/main_cuda.cpp)

add_executable(with_opencv ${WITH_OPENCV})
add_executable(with_cuda ${WITH_CUDA})

target_include_directories(with_opencv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(with_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(with_opencv PRIVATE ${OpenCV_LIBS})
target_link_libraries(with_cuda PRIVATE ${OpenCV_LIBS} CUDA::cudart)

if(ENABLE_QT)
    target_link_libraries(with_opencv PRIVATE Qt6::Widgets)
    target_link_libraries(with_cuda PRIVATE Qt6::Widgets)
    target_compile_definitions(with_opencv PRIVATE WITH_QT)
    target_compile_definitions(with_cuda PRIVATE WITH_QT)
endif()