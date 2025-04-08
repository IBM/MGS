// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef SrcScaledGaussianWeightModifier_H
#define SrcScaledGaussianWeightModifier_H
#include "Lens.h"

#include "CG_SrcScaledGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class SrcScaledGaussianWeightModifier : public CG_SrcScaledGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcScaledGaussianWeightModifier();
      virtual ~SrcScaledGaussianWeightModifier();
      virtual void duplicate(std::unique_ptr<SrcScaledGaussianWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SrcScaledGaussianWeightModifierBase>&& dup) const;

      float _sigma;
      float _max;
};

#endif
