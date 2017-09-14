
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB HDRS ${CMAKE_SOURCE_DIR}/include/*.h*)
file(GLOB SRCS ${CMAKE_SOURCE_DIR}/src/*.c*)

add_library(SDM SHARED ${HDRS} ${SRCS})
target_link_libraries(SDM ${OpenCV_LIBS})