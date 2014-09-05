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
