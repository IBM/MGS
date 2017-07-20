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

#ifndef DstScaledContractedGaussianWeightModifier_H
#define DstScaledContractedGaussianWeightModifier_H

#include "Lens.h"
#include "CG_DstScaledContractedGaussianWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class DstScaledContractedGaussianWeightModifier : public CG_DstScaledContractedGaussianWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, float& contract);
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      DstScaledContractedGaussianWeightModifier();
      virtual ~DstScaledContractedGaussianWeightModifier();
      virtual void duplicate(std::auto_ptr<DstScaledContractedGaussianWeightModifier>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_DstScaledContractedGaussianWeightModifierBase>& dup) const;

      float _sigma, _max, _contract;
};

#endif
