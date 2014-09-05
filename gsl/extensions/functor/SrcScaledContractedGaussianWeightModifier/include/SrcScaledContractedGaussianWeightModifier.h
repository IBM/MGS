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
