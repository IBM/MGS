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

#ifndef SrcRefSumRsqrdInvWeightModifier_H
#define SrcRefSumRsqrdInvWeightModifier_H
#include "Lens.h"

#include "CG_SrcRefSumRsqrdInvWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>
#include <map>

class SrcRefSumRsqrdInvWeightModifier : public CG_SrcRefSumRsqrdInvWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, int& maxDim, bool& setDistance);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcRefSumRsqrdInvWeightModifier();
      virtual ~SrcRefSumRsqrdInvWeightModifier();
      virtual void duplicate(std::unique_ptr<SrcRefSumRsqrdInvWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SrcRefSumRsqrdInvWeightModifierBase>&& dup) const;

      bool _setDistance;
      std::map<float, float> _radiusMap;
      int _maxDistance;
};

#endif
