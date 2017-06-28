cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

ADD_DEFINITIONS(
    -std=c++11 
)

## section: Macro
MACRO(ADD_MSVC_PRECOMPILED_HEADER PrecompiledHeader PrecompiledSource SourcesVar)
  IF(MSVC)
    GET_FILENAME_COMPONENT(PrecompiledBasename ${PrecompiledHeader} NAME_WE)
    SET(PrecompiledBinary "${CMAKE_CURRENT_BINARY_DIR}/${PrecompiledBasename}.pch")
    SET(Sources ${${SourcesVar}})

    SET_SOURCE_FILES_PROPERTIES(${PrecompiledSource}
                                PROPERTIES COMPILE_FLAGS "/Yc\"${PrecompiledHeader}\" /Fp\"${PrecompiledBinary}\""
                                           OBJECT_OUTPUTS "${PrecompiledBinary}")
    SET_SOURCE_FILES_PROPERTIES(${Sources}
                                PROPERTIES COMPILE_FLAGS "/Yu\"${PrecompiledBinary}\" /FI\"${PrecompiledBinary}\" /Fp\"${PrecompiledBinary}\""
                                           OBJECT_DEPENDS "${PrecompiledBinary}")  
    # Add precompiled header to SourcesVar
    LIST(APPEND ${SourcesVar} ${PrecompiledSource})
  ENDIF(MSVC)
ENDMACRO(ADD_MSVC_PRECOMPILED_HEADER)

## start setting
SET (this_target MercuryGestures)
PROJECT(${this_target})


add_definitions(-DUNICODE -D_UNICODE)


## section: include directory
INCLUDE_DIRECTORIES(
  $(OPENCV_DIR)/../../include
  )

## section: source files
# Add your source files here (one file per line), please SORT in alphabetical order for future maintenance
SET (${this_target}_SOURCE_FILES
    ActivityGraph.cpp
	EdgeDetector.cpp
	FaceDetector.cpp
	Hand.cpp
	HandDetector.cpp
	main.cpp
	MovementDetector.cpp
	old.cpp
	SemanticDetector.cpp
	SkinDetector.cpp
	util.cpp
    )

## section: header files
# Add your header files here(one file per line), please SORT in alphabetical order for future maintenance!
SET(${this_target}_HEADER_FILES
    ActivityGraph.h
	EdgeDetector.h
	FaceDetector.h
	HandDetector.h
	MercuryCore.h
	MovementDetector.h
	SemanticDetector.h
	SkinDetector.h
    )

SOURCE_GROUP("Source Files" FILES 
	
	)
SOURCE_GROUP("Resource Files" FILES 
	
	)
SOURCE_GROUP("Header Files" FILES 
	
	)


## section: precompiled header	
#ADD_MSVC_PRECOMPILED_HEADER("precompiled.h" "precompiled.cpp" MySources)
#ADD_LIBRARY(MyLibrary ${MySources})

SET_SOURCE_FILES_PROPERTIES(${this_target}_HEADER_FILES
                            PROPERTIES HEADER_FILE_ONLY TRUE)
LIST(APPEND ${this_target}_SOURCE_FILES ${${this_target}_HEADER_FILES})

## section: add definitions
#   add prefix -D. example> -DSHP
#  - DO NOT add  the following definitions(already defined in ${OSP_DEFINITIONS}:
#   -DSHP, -DWIN32, -D_WINDOWS, -D_DEBUG, -D_USRDLL, -D_CRT_SECURE_NO_DEPRECATE
ADD_DEFINITIONS(
	-DNDEBUG
	-D_CONSOLE
    )

## section: add target

ADD_EXECUTABLE(${this_target} ${${this_target}_SOURCE_FILES})

## section: add dependency
# dependency determines overall build order.
ADD_DEPENDENCIES(${this_target} 
    opencv_calib3d
	opencv_core
	opencv_features2d
	opencv_flann
	opencv_highgui
	opencv_imgcodecs
	opencv_imgproc
	opencv_ml
	opencv_objdetect
	opencv_photo
	opencv_shape
	opencv_stitching
	opencv_superres
#	opencv_ts
	opencv_video
	opencv_videoio
	opencv_videostab
	)

## section: set link libraries
TARGET_LINK_LIBRARIES( ${this_target}
    opencv_calib3d
	opencv_core
	opencv_features2d
	opencv_flann
	opencv_highgui
	opencv_imgcodecs
	opencv_imgproc
	opencv_ml
	opencv_objdetect
	opencv_photo
	opencv_shape
	opencv_stitching
	opencv_superres
#	opencv_ts
	opencv_video
	opencv_videoio
	opencv_videostab
        )
