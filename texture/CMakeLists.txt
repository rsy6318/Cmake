CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(Texture)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(GLUT REQUIRED)

FILE(GLOB_RECURSE CURRENT_HEADERS *.h *.hpp *.cuh)
FILE(GLOB CURRENT_SOURCES *.c *.cpp *.cu)

SET(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)
#SET(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)  
#set(GENCODE ${GENCODE} -gencode=arch=compute_10,code=sm_10)  

# debug options  
SET(CUDA_NVCC_FLAGS_DEBUG "-g -G")  
SET(CMAKE_CXX_FLAGS_DEBUG "-g")  
  
# release options  
SET(CUDA_NVCC_FLAGS_RELEASE "-O2")  
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")  

SET(CUDA_Helper /usr/local/cuda-8.0/samples/common/inc)
SET(CUDA_Helper ${CUDA_Helper} /usr/local/cuda-8.0/samples/common/inc/GL)
SET(CUDA_Helper ${CUDA_Helper} /usr/local/cuda-8.0/samples/common/inc/dynlink)

CUDA_INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} OPTIONS ${GENCODE})
CUDA_INCLUDE_DIRECTORIES(${CUDA_Helper} OPTIONS ${GENCODE})

CUDA_ADD_EXECUTABLE(texture main.cu OPTIONS ${GENCODE})  

TARGET_LINK_LIBRARIES(texture ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY})
