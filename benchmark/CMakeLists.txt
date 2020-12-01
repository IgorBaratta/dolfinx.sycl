cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME demo_helmholtz)
project(${PROJECT_NAME})


find_package(DOLFINX REQUIRED)

# Executable
add_executable(${PROJECT_NAME} main.cpp poisson.c)

# Set C++17 standard
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx)

# Do not throw error for 'multi-line comments' (these are typical in
# rst which includes LaTeX)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment" HAVE_NO_MULTLINE)
set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS "$<$<BOOL:${HAVE_NO_MULTLINE}>:-Wno-comment -Wall -Wextra -pedantic -Werror>")
