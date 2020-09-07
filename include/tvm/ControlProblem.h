/* Copyright 2017-2018 CNRS-AIST JRL and CNRS-UM LIRMM
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <tvm/Task.h>
#include <tvm/graph/CallGraph.h>
#include <tvm/requirements/SolvingRequirements.h>
#include <tvm/scheme/internal/ProblemComputationData.h>
#include <tvm/scheme/internal/ProblemDefinitionEvent.h>
#include <tvm/scheme/internal/ResolutionSchemeBase.h>
#include <tvm/scheme/internal/helpers.h>
#include <tvm/task_dynamics/None.h>
#include <tvm/task_dynamics/abstract/TaskDynamics.h>
#include <tvm/utils/ProtoTask.h>

#include <memory>
#include <vector>

namespace tvm
{
class TVM_DLLAPI TaskWithRequirements : public tvm::internal::ObjWithId
{
public:
  TaskWithRequirements(const Task & task, requirements::SolvingRequirements req);

  Task task;
  requirements::SolvingRequirementsWithCallbacks requirements;
};

using TaskWithRequirementsPtr = std::shared_ptr<TaskWithRequirements>;

class TVM_DLLAPI ControlProblem
{
  friend class LinearizedControlProblem;

public:
  ControlProblem() = default;
  /** \internal We delete these functions because they would require by
   * default a copy of the unique_ptr in the computationData_ map.
   * There is no problem in implementing them as long as there is a real
   * deep copy of the ProblemComputationData. If making this copy, care must
   * be taken that the objects pointed to by the unique_ptr are instances of
   * classes derived from ProblemComputationData.
   */
  ControlProblem(const ControlProblem &) = delete;
  ControlProblem & operator=(const ControlProblem &) = delete;

  TaskWithRequirementsPtr add(const Task & task, const requirements::SolvingRequirements & req = {});
  template<constraint::Type T>
  TaskWithRequirementsPtr add(utils::ProtoTask<T> proto,
                              const task_dynamics::abstract::TaskDynamics & td,
                              const requirements::SolvingRequirements & req = {});
  template<constraint::Type T>
  TaskWithRequirementsPtr add(utils::LinearProtoTask<T> proto,
                              const task_dynamics::abstract::TaskDynamics & td,
                              const requirements::SolvingRequirements & req = {});
  template<constraint::Type T>
  TaskWithRequirementsPtr add(utils::LinearProtoTask<T> proto, const requirements::SolvingRequirements & req = {});
  void add(TaskWithRequirementsPtr tr);
  void remove(TaskWithRequirements * tr);
  const std::vector<TaskWithRequirementsPtr> & tasks() const;

  /** Number of tasks in the problem.*/
  int size() const;

  /** Compute all quantities necessary for solving the problem.*/
  void update();

  /** Finalize the data of the solver*/
  void finalize();

protected:
  class Updater
  {
  public:
    Updater();

    template<typename T, typename... Args>
    void addInput(std::shared_ptr<T> source, Args... args);
    template<typename T>
    void removeInput(T * source);

    /** Manually calls for an update of the call graph, if needed.*/
    void refresh();
    /** Execute the call graph.*/
    void run();

  private:
    bool upToDate_;
    std::shared_ptr<graph::internal::Inputs> inputs_;
    graph::CallGraph updateGraph_;
  };

  virtual void update_() {}
  virtual void finalize_() {}

  Updater updater_;

private:
  void notify(const scheme::internal::ProblemDefinitionEvent & e);
  void addCallBackToTask(TaskWithRequirementsPtr tr);

  // Note: we want to keep the tasks in the order they were introduced, mostly
  // for human understanding and debugging purposes, so that we take a
  // std::vector.
  // If this induces too much overhead when adding/removing a constraint, then
  // we should consider std::set.
  std::vector<TaskWithRequirementsPtr> tr_;

  // Tokens to identify and keep callbacks alive
  tvm::utils::internal::map<TaskWithRequirements *, std::vector<internal::PairElementToken>> callbackTokens_;

  // Computation data for the resolution schemes
  std::map<scheme::identifier, std::unique_ptr<scheme::internal::ProblemComputationData>> computationData_;

  bool finalized_ = false;

  template<typename Problem, typename Scheme>
  friend scheme::internal::ProblemComputationData * scheme::internal::getComputationData(
      Problem & problem,
      const Scheme & resolutionScheme);
};

template<constraint::Type T>
TaskWithRequirementsPtr ControlProblem::add(utils::ProtoTask<T> proto,
                                            const task_dynamics::abstract::TaskDynamics & td,
                                            const requirements::SolvingRequirements & req)
{
  return add({proto, td}, req);
}

template<constraint::Type T>
TaskWithRequirementsPtr ControlProblem::add(utils::LinearProtoTask<T> proto,
                                            const task_dynamics::abstract::TaskDynamics & td,
                                            const requirements::SolvingRequirements & req)
{
  return add({proto, td}, req);
}

template<constraint::Type T>
TaskWithRequirementsPtr ControlProblem::add(utils::LinearProtoTask<T> proto,
                                            const requirements::SolvingRequirements & req)
{
  return add({proto, task_dynamics::None()}, req);
}

template<typename T, typename... Args>
inline void ControlProblem::Updater::addInput(std::shared_ptr<T> source, Args... args)
{
  inputs_->addInput(source, args...);
  upToDate_ = false;
}

template<typename T>
inline void ControlProblem::Updater::removeInput(T * source)
{
  inputs_->removeInput(source);
  upToDate_ = false;
}
} // namespace tvm
