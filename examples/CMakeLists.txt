
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SRCS ${CMAKE_SOURCE_DIR}/examples/*.c*)

add_executable(example ${SRCS})
target_link_libraries(example ${OpenCV_LIBS} SDM)