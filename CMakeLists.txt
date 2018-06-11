cmake_minimum_required(VERSION 2.10)
project(gimp-cuda-plugin)

find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)

set(CMAKE_BUILD_TYPE Release CACHE PATH "default build type" FORCE)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(APPLE)
        set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/Library/Application\ Support/GIMP/2.8/plug-ins"
            CACHE PATH "default plugin installation path" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.gimp-2.8/plug-ins"
            CACHE PATH "default plugin installation path" FORCE)
    endif()
endif()

pkg_check_modules(GIMPUI gimpui-2.0 REQUIRED)
include_directories(${GIMPUI_INCLUDE_DIRS})

cuda_add_executable(multi_res_filter multi_res_host.cu gimp_gui.cpp gimp_main.cpp multi_res_cpu.cpp)
target_link_libraries(multi_res_filter ${GIMPUI_LIBRARIES})

install(TARGETS multi_res_filter RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
