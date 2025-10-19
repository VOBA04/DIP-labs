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
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    find_package(VTK REQUIRED)
    message(STATUS "Qt support: ENABLED")
else()
    message(STATUS "Qt support: DISABLED")
endif()

add_library(dip_core STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image.cpp
)
target_include_directories(dip_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

add_executable(with_opencv ${CMAKE_CURRENT_SOURCE_DIR}/src/mains/main.cpp)
add_executable(train ${CMAKE_CURRENT_SOURCE_DIR}/src/mains/train.cpp)
add_executable(test_model ${CMAKE_CURRENT_SOURCE_DIR}/src/mains/test_model.cpp)

target_link_libraries(with_opencv PRIVATE 
    dip_core
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

target_link_libraries(train PRIVATE 
    dip_core
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

target_link_libraries(test_model PRIVATE 
    dip_core
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

if(ENABLE_QT)
    target_link_libraries(with_opencv PRIVATE Qt5::Widgets ${VTK_LIBRARIES})
    target_link_libraries(test_model PRIVATE Qt5::Widgets)
    target_compile_definitions(with_opencv PRIVATE WITH_QT)
    target_compile_definitions(test_model PRIVATE WITH_QT)
    target_link_libraries(train PRIVATE Qt5::Widgets)
    target_compile_definitions(train PRIVATE WITH_QT)
endif()