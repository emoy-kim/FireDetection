cmake_minimum_required(VERSION 3.8)

project("FireDetection")

include(cmake/check-compiler.cmake)

set(CMAKE_CXX_STANDARD 17)

set(
	SOURCE_FILES 
		main.cpp
		source/RChannelFireDetection.cpp
		source/TrainedColorFireDetection.cpp
		source/CovarianceFireDetection.cpp
		source/FlowRateFireDetection.cpp
		source/UnitedFireDetection.cpp
)

configure_file(include/ProjectPath.h.in ${PROJECT_BINARY_DIR}/ProjectPath.h @ONLY)

include_directories("include")
if(MSVC)
   include(cmake/add-libraries-windows.cmake)
else()
   include(cmake/add-libraries-linux.cmake)
endif()

add_executable(FireDetection ${SOURCE_FILES})

if(MSVC)
   include(cmake/target-link-libraries-windows.cmake)
else()
   include(cmake/target-link-libraries-linux.cmake)
endif()

target_include_directories(FireDetection PUBLIC ${CMAKE_BINARY_DIR})