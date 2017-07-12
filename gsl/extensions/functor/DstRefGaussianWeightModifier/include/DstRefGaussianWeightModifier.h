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

#ifndef DstRefGaussianWeightModifier_H
#define DstRefGaussianWeightModifier_H
#include "Lens.h"

#include "CG_DstRefGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class DstRefGaussianWeightModifier : public CG_DstRefGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, int& wrapDistance);
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      DstRefGaussianWeightModifier();
      virtual ~DstRefGaussianWeightModifier();
      virtual void duplicate(std::auto_ptr<DstRefGaussianWeightModifier>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_DstRefGaussianWeightModifierBase>& dup) const;

      float _sigma;
      float _max;
      int _wrapDistance;
};

#endif
