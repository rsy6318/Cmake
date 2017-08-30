cmake_minimum_required(VERSION 2.8)


project(tex)


find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB_RECURSE CURRENT_HEADERS *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES *.c *.cpp *.cu)

# build options  
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35)  
#set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)  
#set(GENCODE ${GENCODE} -gencode=arch=compute_10,code=sm_10)  


#message(STATUS "OpenCV library status:")
#message(STATUS "    version: ${OpenCV_VERSION}")
#message(STATUS "    Opencv libraries: ${OpenCV_LIBS}")
#message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
#message(STATUS "OpenGL library status:")
#message(STATUS "    OpenGL libraries: ${OPENGL_LIBRARIES}")

set(CUDA_Helper /usr/local/cuda-8.0/samples/common/inc)
set(CUDA_Helper ${CUDA_Helper} /usr/local/cuda-8.0/samples/common/inc/GL)
set(CUDA_Helper ${CUDA_Helper} /usr/local/cuda-8.0/samples/common/inc/dynlink)

#message(STATUS "helper direction: ${CUDA_Helper}")

cuda_include_directories(${OpenCV_INCLUDE_DIRS} OPTIONS ${GENCODE})
cuda_include_directories(${CUDA_Helper} OPTIONS ${GENCODE})

cuda_add_executable(tex main.cu OPTIONS ${GENCODE})

target_link_libraries(tex ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY})
