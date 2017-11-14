set(TVM_SOURCES
  Clock.cpp
  ControlProblem.cpp
  LinearizedControlProblem.cpp
  Robot.cpp
  Space.cpp
  Task.cpp
  Variable.cpp
  VariableVector.cpp

  constraint/BasicLinearConstraint.cpp
  constraint/ConstraintBase.cpp
  constraint/Constraint.cpp
  constraint/LinearConstraint.cpp
  constraint/LinearizedTaskConstraint.cpp

  event/Listener.cpp
  event/Source.cpp

  function/BasicLinearFunction.cpp
  function/Function.cpp
  function/IdentityFunction.cpp
  function/LinearFunction.cpp

  graph/CallGraph.cpp
  graph/Inputs.cpp

  internal/FirstOrderProvider.cpp
  internal/MatrixProperties.cpp

  requirements/SolvingRequirements.cpp

  scheme/Assignment.cpp
  scheme/AssignmentTarget.cpp
  scheme/ResolutionScheme.cpp
  scheme/SchemeAbilities.cpp
  scheme/WeightedLeastSquares.cpp

  task_dynamics/None.cpp
  task_dynamics/Proportional.cpp
  task_dynamics/ProportionalDerivative.cpp
  task_dynamics/TaskDynamics.cpp
)

set(TVM_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tvm)
set(TVM_HEADERS
  ${TVM_INCLUDE_DIR}/api.h
  ${TVM_INCLUDE_DIR}/Clock.h
  ${TVM_INCLUDE_DIR}/ControlProblem.h
  ${TVM_INCLUDE_DIR}/defs.h
  ${TVM_INCLUDE_DIR}/LinearizedControlProblem.h
  ${TVM_INCLUDE_DIR}/Range.h
  ${TVM_INCLUDE_DIR}/Robot.h
  ${TVM_INCLUDE_DIR}/Space.h
  ${TVM_INCLUDE_DIR}/Task.h
  ${TVM_INCLUDE_DIR}/Variable.h
  ${TVM_INCLUDE_DIR}/VariableVector.h

  ${TVM_INCLUDE_DIR}/constraint/abstract/Constraint.h
  ${TVM_INCLUDE_DIR}/constraint/abstract/LinearConstraint.h
  ${TVM_INCLUDE_DIR}/constraint/BasicLinearConstraint.h
  ${TVM_INCLUDE_DIR}/constraint/enums.h
  ${TVM_INCLUDE_DIR}/constraint/internal/ConstraintBase.h
  ${TVM_INCLUDE_DIR}/constraint/internal/LinearizedTaskConstraint.h

  ${TVM_INCLUDE_DIR}/event/enums.h
  ${TVM_INCLUDE_DIR}/event/Listener.h
  ${TVM_INCLUDE_DIR}/event/Source.h

  ${TVM_INCLUDE_DIR}/exception/exceptions.h

  ${TVM_INCLUDE_DIR}/function/abstract/Function.h
  ${TVM_INCLUDE_DIR}/function/abstract/LinearFunction.h
  ${TVM_INCLUDE_DIR}/function/BasicLinearFunction.h
  ${TVM_INCLUDE_DIR}/function/IdentityFunction.h

  ${TVM_INCLUDE_DIR}/graph/abstract/Node.h
  ${TVM_INCLUDE_DIR}/graph/abstract/Node.hpp
  ${TVM_INCLUDE_DIR}/graph/abstract/OutputSelector.h
  ${TVM_INCLUDE_DIR}/graph/abstract/Outputs.h
  ${TVM_INCLUDE_DIR}/graph/CallGraph.h
  ${TVM_INCLUDE_DIR}/graph/internal/AbstractNode.h
  ${TVM_INCLUDE_DIR}/graph/internal/Inputs.h
  ${TVM_INCLUDE_DIR}/graph/internal/Inputs.hpp

  ${TVM_INCLUDE_DIR}/internal/enums.h
  ${TVM_INCLUDE_DIR}/internal/FirstOrderProvider.h
  ${TVM_INCLUDE_DIR}/internal/MatrixProperties.h
  ${TVM_INCLUDE_DIR}/internal/MatrixWithProperties.h

  ${TVM_INCLUDE_DIR}/requirements/abstract/SingleSolvingRequirement.h
  ${TVM_INCLUDE_DIR}/requirements/AnisotropicWeight.h
  ${TVM_INCLUDE_DIR}/requirements/PriorityLevel.h
  ${TVM_INCLUDE_DIR}/requirements/SolvingRequirements.h
  ${TVM_INCLUDE_DIR}/requirements/ViolationEvaluation.h
  ${TVM_INCLUDE_DIR}/requirements/Weight.h

  ${TVM_INCLUDE_DIR}/scheme/abstract/ResolutionScheme.h
  ${TVM_INCLUDE_DIR}/scheme/internal/Assignment.h
  ${TVM_INCLUDE_DIR}/scheme/internal/AssignmentTarget.h
  ${TVM_INCLUDE_DIR}/scheme/internal/CompiledAssignment.h
  ${TVM_INCLUDE_DIR}/scheme/internal/CompiledAssignmentWrapper.h
  ${TVM_INCLUDE_DIR}/scheme/internal/SchemeAbilities.h
  ${TVM_INCLUDE_DIR}/scheme/WeightedLeastSquares.h

  ${TVM_INCLUDE_DIR}/task_dynamics/abstract/TaskDynamics.h
  ${TVM_INCLUDE_DIR}/task_dynamics/enums.h
  ${TVM_INCLUDE_DIR}/task_dynamics/None.h
  ${TVM_INCLUDE_DIR}/task_dynamics/ProportionalDerivative.h
  ${TVM_INCLUDE_DIR}/task_dynamics/Proportional.h
)

#Helper functions to assign files to groups in IDE, mimicking the organization of files on disk
#Adapted from https://stackoverflow.com/a/31423421
function(assign_source_group folder)
  foreach(_source IN ITEMS ${ARGN})
    if (IS_ABSOLUTE "${_source}")
        file(RELATIVE_PATH _source_rel "${CMAKE_SOURCE_DIR}" "${_source}")
    else()
        set(_source_rel "${_source}")
    endif()
    #remove include/tvm/ if present in the path
    string(REPLACE "include/tvm/" "" _source_rel_modified "${_source_rel}")
    get_filename_component(_source_path "${_source_rel_modified}" PATH)
    string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
    source_group("${folder}\\${_source_path_msvc}" FILES "${_source}")
  endforeach()
endfunction(assign_source_group)

assign_source_group("sources" ${TVM_SOURCES})
assign_source_group("headers" ${TVM_HEADERS})

add_library(TVM SHARED ${TVM_SOURCES} ${TVM_HEADERS})
set_target_properties(TVM PROPERTIES COMPILE_FLAGS "-DTVM_EXPORTS")

install(TARGETS TVM
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(DIRECTORY ${TVM_INCLUDE_DIR} DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")