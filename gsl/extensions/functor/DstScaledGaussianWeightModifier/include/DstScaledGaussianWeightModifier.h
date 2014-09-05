// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef DstScaledGaussianWeightModifier_H
#define DstScaledGaussianWeightModifier_H
#include "Lens.h"

#include "CG_DstScaledGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class DstScaledGaussianWeightModifier : public CG_DstScaledGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max);
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      DstScaledGaussianWeightModifier();
      virtual ~DstScaledGaussianWeightModifier();
      virtual void duplicate(std::auto_ptr<DstScaledGaussianWeightModifier>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_DstScaledGaussianWeightModifierBase>& dup) const;

      float _sigma;
      float _max;
};

#endif
