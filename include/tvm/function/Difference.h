/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#pragma once

#include <tvm/function/abstract/Function.h>
#include <tvm/graph/abstract/OutputSelector.h>

namespace tvm
{

namespace function
{

/** This class implements a function that computes the difference between two functions
 *
 * - value is BinaryOp{}(lhs.value() - rhs.value())
 * - velocity is lhs.velocity() - rhs.velocity()
 * - normalAcceleration is lhs.normalAcceleration() - rhs.normalAcceleration()
 * - jacobian(x) is lhs.jacobian(x) - rhs.jacobian(x) (if lhs or rhs does not depend on x then it is not included)
 * - jDot(x) is lhs.jDot(x) - rhs.jDot(x) (same remark as the jacobian)
 */
template<typename BinaryOp = std::minus<Eigen::VectorXd>>
class TVM_DLLAPI Difference : public graph::abstract::OutputSelector<function::abstract::Function>
{
public:
  SET_UPDATES(Difference, Value, Velocity, Jacobian, NormalAcceleration, JDot)

  /** Constructor
   *
   * \param lhs Left-hand side argument of the difference operator
   *
   * \param rhs Right-hand side argument of the difference operator
   *
   */
  Difference(FunctionPtr lhs, FunctionPtr rhs)
  : graph::abstract::OutputSelector<function::abstract::Function>(lhs->size()), lhs_(lhs), rhs_(rhs)
  {
    static_assert(std::is_convertible_v<decltype(BinaryOp{}(lhs->value(), rhs->value())), Eigen::VectorXd>, "The BinaryOp parameter in Difference must return an Eigen::VectorXd convertible object");
    if(lhs_->size() != rhs_->size())
    {
      throw std::runtime_error("[Difference::Difference] lhs and rhs must have the same size");
    }
    processOutput(Output::Value, Update::Value, &Difference::updateValue, Output::Value);
    processOutput(Output::Jacobian, Update::Jacobian, &Difference::updateJacobian, Output::Jacobian);
    processOutput(Output::Velocity, Update::Velocity, &Difference::updateVelocity, Output::Velocity);
    processOutput(Output::NormalAcceleration, Update::NormalAcc, &Difference::updateNormalAcceleration, Output::NormalAcceleration);
    processOutput(Output::JDot, Update::JDot, &Difference::updateJDot, Output::JDot);
    for (const auto& xi : lhs_->variables())
    {
      bool lin = lhs_->linearIn(*xi) && (!rhs_->variables().contains(*xi) || rhs_->linearIn(*xi));
      addVariable(xi, lin);
    }
    for (const auto& xi : rhs_->variables())
    {
      bool lin = rhs_->linearIn(*xi) && (!lhs_->variables().contains(*xi) || lhs_->linearIn(*xi));
      addVariable(xi, lin);
    }
  }

protected:
  template<typename Out, typename Up, typename... In>
  void processOutput(Out output, Up u, void (Difference::* update)(), In... inputs)
  {
    // We enable the output is all the required inputs are enabled for g and h
    bool enableOutput = (... && (lhs_->isOutputEnabled(inputs) && rhs_->isOutputEnabled(inputs)));

    if (enableOutput)
    {
      registerUpdates(u, update);
      addInputDependency(u, lhs_, inputs...);
      addInputDependency(u, rhs_, inputs...);
      addOutputDependency(output, u);
    }
    else
      disableOutput(output);
  }

  void updateValue()
  {
    value_ = BinaryOp{}(lhs_->value(), rhs_->value());
  }
  void updateVelocity()
  {
    velocity_ = lhs_->velocity() - rhs_->velocity();
  }
  void updateJacobian()
  {
    for(const auto & xi : rhs_->variables())
    {
      jacobian_.at(xi.get()).setZero();
    }
    for(const auto & xi : lhs_->variables())
    {
      jacobian_.at(xi.get()) = lhs_->jacobian(*xi);
    }
    for(const auto & xi : rhs_->variables())
    {
      jacobian_.at(xi.get()) -= rhs_->jacobian(*xi);
    }
  }
  void updateNormalAcceleration()
  {
    normalAcceleration_ = lhs_->normalAcceleration() - rhs_->normalAcceleration();
  }
  void updateJDot()
  {
    for(const auto & xi : rhs_->variables())
    {
      JDot_.at(xi.get()).setZero();
    }
    for(const auto & xi : lhs_->variables())
    {
      JDot_.at(xi.get()) = lhs_->JDot(*xi);
    }
    for(const auto & xi : rhs_->variables())
    {
      JDot_.at(xi.get()) -= rhs_->JDot(*xi);
    }
  }

  FunctionPtr lhs_;
  FunctionPtr rhs_;
};

} // namespace function

} // namespace tvm
