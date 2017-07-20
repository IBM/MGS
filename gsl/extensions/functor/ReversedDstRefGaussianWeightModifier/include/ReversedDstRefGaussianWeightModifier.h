// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef ReversedDstRefGaussianWeightModifier_H
#define ReversedDstRefGaussianWeightModifier_H
#include "Lens.h"

#include "CG_ReversedDstRefGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class ReversedDstRefGaussianWeightModifier : public CG_ReversedDstRefGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, int& wrapDistance);
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      ReversedDstRefGaussianWeightModifier();
      virtual ~ReversedDstRefGaussianWeightModifier();
      virtual void duplicate(std::auto_ptr<ReversedDstRefGaussianWeightModifier>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ReversedDstRefGaussianWeightModifierBase>& dup) const;

      float _sigma;
      float _max;
      int _wrapDistance;
};

#endif
