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

#include <tvm/constraint/enums.h>
#include <tvm/constraint/internal/RHSVectors.h>
#include <tvm/internal/FirstOrderProvider.h>
#include <tvm/graph/abstract/OutputSelector.h>

#include <Eigen/Core>

#include <memory>

namespace tvm
{

namespace constraint
{

namespace abstract
{

  /** Base class for representing a constraint.
    *
    * It manages the enabling/disabling of the outputs L, U and E (depending
    * on its type and rhs convention).
    *
    * FIXME: have the updateValue here and add an output check()
    *
    * \dot
    * digraph "update graph" {
    *   rankdir="LR";
    *   {
    *     rank = same; node [shape=hexagon];
    *     Value; Jacobian; L; U; E;
    *   }
    *   {
    *     rank = same; node [style=invis, label=""];
    *     outValue; outJacobian; outL; outU; outE;
    *   }
    *   Value -> outValue [label="value()"];
    *   Jacobian -> outJacobian [label="jacobian(x_i)"];
    *   L -> outL [label="l()"];
    *   U -> outU [label="u()"];
    *   E -> outE [label="e()"];
    * }
    * \enddot
    */
  class TVM_DLLAPI Constraint : public tvm::internal::ObjWithId, public graph::abstract::OutputSelector<Constraint, tvm::internal::FirstOrderProvider>
  {
  public:
    SET_OUTPUTS(Constraint, L, U, E)

    /** \internal by default, these methods return the cached value.
      * However, they are virtual in case the user might want to bypass the cache.
      * This would be typically the case if he/she wants to directly return the
      * output of another method.
      */
    /** Return the vector \p l
      * \warning the call is valid only if \p l exists for the given constraint
      * conventions, but the method does not throw if it is not the case.
      */
    virtual const Eigen::VectorXd& l() const;
    /** Return the vector \p u
      * \warning the call is valid only if \p l exists for the given constraint
      * conventions, but the method does not throw if it is not the case.
      */
    virtual const Eigen::VectorXd& u() const;
    /** Return the vector \p e
      * \warning the call is valid only if \p l exists for the given constraint
      * conventions, but the method does not throw if it is not the case.
      */
    virtual const Eigen::VectorXd& e() const;

    /** Return the type of the constraint.*/
    Type type() const;

    /** Check whether this is an equality constraint. */
    bool isEquality() const;

    /** Return the convention for the right-hand side \p e, \p l, \p u or both
      * \p l and \p u of the constraint.
      */
    RHS rhs() const;

  protected:
    /** Constructor. Only available to derived classes.
      * \param ct The constraint type
      * \param cr The rhs convention
      * \param The (output) size of the constraint
      */
    Constraint(Type ct, RHS cr, int m=0);

    /** Resize the cache (rhs vector(s), jacobian matrices,...) for the current
      * size of the constraint.
      */
    void resizeCache() override;

    /** Direct (non-const) access to \p l for derived classes */
    Eigen::VectorXd& lRef();
    /** Direct (non-const) access to \p u for derived classes */
    Eigen::VectorXd& uRef();
    /** Direct (non-const) access to \p e for derived classes */
    Eigen::VectorXd& eRef();

    /** Cache for l, u and e */
    internal::RHSVectors vectors_;

  private:
    Type  cstrType_;      // The constraint type
    RHS   constraintRhs_; // The rhs convention
  };


  inline const Eigen::VectorXd& Constraint::l() const
  {
    return vectors_.l();
  }

  inline const Eigen::VectorXd& Constraint::u() const
  {
    return vectors_.u();
  }

  inline const Eigen::VectorXd& Constraint::e() const
  {
    return vectors_.e();
  }

  inline Eigen::VectorXd& Constraint::lRef()
  {
    return vectors_.l();
  }

  inline Eigen::VectorXd& Constraint::uRef()
  {
    return vectors_.u();
  }

  inline Eigen::VectorXd& Constraint::eRef()
  {
    return vectors_.e();
  }

}  // namespace abstract

}  // namespace constraint

}  // namespace tvm
