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

#ifndef SrcRefGaussianWeightModifier_H
#define SrcRefGaussianWeightModifier_H
#include "Lens.h"

#include "CG_SrcRefGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class SrcRefGaussianWeightModifier : public CG_SrcRefGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, int& wrapDistance);
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcRefGaussianWeightModifier();
      virtual ~SrcRefGaussianWeightModifier();
      virtual void duplicate(std::auto_ptr<SrcRefGaussianWeightModifier>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_SrcRefGaussianWeightModifierBase>& dup) const;

      float _sigma;
      float _max;
      int _wrapDistance;
};

#endif
