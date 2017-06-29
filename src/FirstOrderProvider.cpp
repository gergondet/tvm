#include "errors.h"
#include "FirstOrderProvider.h"
#include "Variable.h"

namespace taskvm
{
  namespace internal
  {
    const Eigen::VectorXd& FirstOrderProvider::value() const
    {
      if (hasOutput(Output::Value))
        return valueNoCheck();
      else
        throw UnusedOutput(/*description*/); //TODO add description of the error
    }

    const Eigen::MatrixXd& FirstOrderProvider::jacobian(const Variable& x) const
    {
      if (hasOutput(Output::Jacobian))
        return jacobianNoCheck(x);
      else
        throw UnusedOutput(/*description*/); //TODO add description of the error
    }

    const Eigen::VectorXd& FirstOrderProvider::valueNoCheck() const
    {
      return value_;
    }

    const Eigen::MatrixXd& FirstOrderProvider::jacobianNoCheck(const Variable& x) const
    {
      return jacobian_.at(&x);
    }

    int FirstOrderProvider::size() const
    {
      return m_;
    }

    const std::vector<std::shared_ptr<Variable>>& FirstOrderProvider::variables() const
    {
      return variables_;
    }

    void FirstOrderProvider::resizeCache()
    {
      if (hasOutput(Output::Value))
        value_.resize(m_);

      if (hasOutput(Output::Jacobian))
      {
        for (auto v : variables_)
          jacobian_[v.get()].resize(m_, v->size());
      }
    }

    void FirstOrderProvider::addVariable(std::shared_ptr<Variable> v)
    {
      if (std::find(variables_.begin(), variables_.end(), v) == variables_.end())
        variables_.push_back(v);
      else
        throw DuplicateVariable(/*desc*/); //TODO

      jacobian_[v.get()].resize(m_, v->size());

      addVariable_(v);
    }

    void FirstOrderProvider::removeVariable(std::shared_ptr<Variable> v)
    {
      auto it = std::find(variables_.begin(), variables_.end(), v);
      if (it == variables_.end())
        throw NonExistingVariable(/*desc*/); //TODO
      else
      {
        variables_.erase(it);
        jacobian_.erase(v.get());
      }
    }

    void FirstOrderProvider::addVariable_(std::shared_ptr<Variable>)
    {
      //do nothing
    }

    void FirstOrderProvider::removeVariable_(std::shared_ptr<Variable>)
    {
      //do nothing
    }
  }
}