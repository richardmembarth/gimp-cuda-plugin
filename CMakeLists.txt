CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(gimp-cuda-plugin)

INCLUDE(FindPkgConfig)
FIND_PACKAGE(CUDA REQUIRED)

SET(CMAKE_BUILD_TYPE Release CACHE PATH "default build type" FORCE)
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    IF(APPLE)
        SET(CMAKE_INSTALL_PREFIX "$ENV{HOME}/Library/Application\ Support/GIMP/2.8/plug-ins"
            CACHE PATH "default plugin installation path" FORCE)
    ELSE()
        SET(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.gimp-2.8/plug-ins"
            CACHE PATH "default plugin installation path" FORCE)
    ENDIF()
ENDIF()

pkg_check_modules(GIMPUI gimpui-2.0 REQUIRED)
include_directories(${GIMPUI_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(multi_res_filter multi_res_host.cu gimp_gui.cpp gimp_main.cpp multi_res_cpu.cpp)
TARGET_LINK_LIBRARIES(multi_res_filter ${GIMPUI_LIBRARIES})

INSTALL(TARGETS multi_res_filter RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
