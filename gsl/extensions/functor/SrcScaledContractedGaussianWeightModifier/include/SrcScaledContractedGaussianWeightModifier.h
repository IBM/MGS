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

#ifndef SrcScaledContractedGaussianWeightModifier_H
#define SrcScaledContractedGaussianWeightModifier_H

#include "Lens.h"
#include "CG_SrcScaledContractedGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class SrcScaledContractedGaussianWeightModifier : public CG_SrcScaledContractedGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, float& contract);
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcScaledContractedGaussianWeightModifier();
      virtual ~SrcScaledContractedGaussianWeightModifier();
      virtual void duplicate(std::auto_ptr<SrcScaledContractedGaussianWeightModifier>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_SrcScaledContractedGaussianWeightModifierBase>& dup) const;

      float _sigma, _max, _contract;
};

#endif
