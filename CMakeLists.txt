cmake_minimum_required(VERSION 3.18)

project(DIP_Lab LANGUAGES C CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_definitions(-DEXTERNAL="${CMAKE_SOURCE_DIR}/external")
add_definitions(-DMODELS="${CMAKE_SOURCE_DIR}/models")

# Add option to enable/disable Qt usage
option(ENABLE_QT "Enable Qt Widgets for file dialog" ON)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

if(ENABLE_QT)
# Use Qt5 to match VTK's Qt dependency (apt libvtk9-dev commonly uses Qt5)
find_package(Qt5 COMPONENTS Widgets REQUIRED)
    find_package(VTK REQUIRED)
    message(STATUS "Qt support: ENABLED")
else()
    message(STATUS "Qt support: DISABLED")
endif()

file(GLOB CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

set(WITH_OPENCV ${CPP_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/mains/main.cpp)

add_executable(with_opencv ${WITH_OPENCV})
add_executable(train ${CMAKE_CURRENT_SOURCE_DIR}/src/mains/train.cpp)

target_include_directories(with_opencv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_include_directories(train PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(with_opencv PRIVATE 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

target_link_libraries(train PRIVATE 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

if(ENABLE_QT)
    target_link_libraries(with_opencv PRIVATE Qt5::Widgets ${VTK_LIBRARIES})
    target_compile_definitions(with_opencv PRIVATE WITH_QT)
endif()