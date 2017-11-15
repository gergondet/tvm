#include <tvm/scheme/WeightedLeastSquares.h>

#include <tvm/LinearizedControlProblem.h>
#include <tvm/constraint/internal/LinearizedTaskConstraint.h>
#include <tvm/constraint/abstract/LinearConstraint.h>

#include <iostream>

namespace tvm
{

namespace scheme
{
  using namespace internal;
  using VET = requirements::ViolationEvaluationType;

  WeightedLeastSquares::WeightedLeastSquares(double scalarizationWeight)
    : LinearResolutionScheme<WeightedLeastSquares>({ 2, {{0, {true, {VET::L2}}}, {1,{false, {VET::L2}}}}, true })
    , scalarizationWeight_(scalarizationWeight)
  {
  }

  void WeightedLeastSquares::solve_(LinearizedControlProblem& problem, Memory& memory) const
  {
    for (auto& a : memory.assignments)
      a.run();

    std::cout << "A =\n" << memory.A << std::endl;
    std::cout << "b = " << memory.b.transpose() << std::endl;
    std::cout << "C =\n" << memory.C << std::endl;
    std::cout << "l = " << memory.l.transpose() << std::endl;
    std::cout << "u = " << memory.u.transpose() << std::endl;

    bool b = memory.ls.solve(memory.A, memory.b, memory.C, memory.l, memory.u);
    std::cout << memory.ls.inform() << std::endl;
    std::cout << memory.ls.result().transpose() << std::endl;
  }

  std::unique_ptr<WeightedLeastSquares::Memory> WeightedLeastSquares::createComputationData_(const LinearizedControlProblem& problem) const
  {
    auto memory = std::unique_ptr<Memory>(new Memory(id()));
    const auto& constraints = problem.constraints();

    //scanning constraints
    int m0 = 0;
    int m1 = 0;
    for (auto c : constraints)
    {
      abilities_.check(c.constraint, c.requirements); //FIXME: should be done in a parent class
      memory->addVariable(c.constraint->variables()); //FIXME: should be done in a parent class

      if (c.requirements->priorityLevel().value() == 0)
        m0 += c.constraint->size();
      else
        m1 += c.constraint->size();  //note: we cannot have double sided constraints at this level.
    }

    if (m1 == 0)
      m1 = memory->variables().size();

    //allocating memory for the solver
    memory->resize(m0, m1, big_number_);

    //assigments
    m0 = 0;
    m1 = 0;
    const auto& x = memory->variables();
    for (auto c : constraints)
    {
      int p = c.requirements->priorityLevel().value();
      if (p == 0)
      {
        RangePtr r = std::make_shared<Range>(m0, c.constraint->size()); //FIXME: for now we do not keep a pointer on the range nor the target.
        AssignmentTarget target(r, { memory->basePtr, &memory->C }, { memory->basePtr, &memory->l }, { memory->basePtr, &memory->u }, constraint::RHS::AS_GIVEN, x.size());
        memory->assignments.emplace_back(Assignment(c.constraint, c.requirements, target, x));
        m0 += c.constraint->size();
      }
      else
      {
        RangePtr r = std::make_shared<Range>(m1, c.constraint->size()); //FIXME: for now we do not keep a pointer on the range nor the target.
        AssignmentTarget target(r, { memory->basePtr, &memory->A }, { memory->basePtr, &memory->b }, constraint::Type::EQUAL, constraint::RHS::AS_GIVEN);
        memory->assignments.emplace_back(Assignment(c.constraint, c.requirements, target, x, std::pow(scalarizationWeight_, p - 1)));
        m1 += c.constraint->size();
      }
    }
    if (m1 == 0)
    {
      memory->A.setIdentity();
      memory->b.setZero();
    }

    return memory;
  }

  WeightedLeastSquares::Memory::Memory(int solverId)
    : ProblemComputationData(solverId)
    , basePtr(new int)
  {
  }

  void WeightedLeastSquares::Memory::resize(int m0, int m1, double big_number)
  {
    int n = x_.size();
    A.resize(m1, n);
    A.setZero();
    C.resize(m0, n);
    C.setZero();
    b.resize(m1);
    b.setZero();
    l = Eigen::VectorXd::Constant(m0 + n, -big_number);
    u = Eigen::VectorXd::Constant(m0 + n, +big_number);
    ls.resize(n, m0, Eigen::lssol::eType::LS1);
  }

}  // namespace scheme

}  // namespace tvm
