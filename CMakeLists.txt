# Use 'add_subdirectory(openevo)' to add this library to your project's CMakeLists.txt.
# Use 'target_link_libraries(<target> openevo)' to link with your compilation target.
find_package( OpenCV REQUIRED )
add_library(openevo STATIC evo.cpp)
target_link_libraries(openevo ${OpenCV_LIBS})
target_include_directories(openevo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_definitions(openevo PRIVATE -DDEBUG -DUSE_PROFILER=1)
endif()

