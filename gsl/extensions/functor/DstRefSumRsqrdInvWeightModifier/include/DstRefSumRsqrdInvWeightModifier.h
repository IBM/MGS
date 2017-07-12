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

#ifndef DstRefSumRsqrdInvWeightModifier_H
#define DstRefSumRsqrdInvWeightModifier_H
#include "Lens.h"

#include "CG_DstRefSumRsqrdInvWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>
#include <map>

class DstRefSumRsqrdInvWeightModifier : public CG_DstRefSumRsqrdInvWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, int& maxDim, bool& setDistance);
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      DstRefSumRsqrdInvWeightModifier();
      virtual ~DstRefSumRsqrdInvWeightModifier();
      virtual void duplicate(std::auto_ptr<DstRefSumRsqrdInvWeightModifier>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_DstRefSumRsqrdInvWeightModifierBase>& dup) const;

      bool _setDistance;
      std::map<float, float> _radiusMap;
      int _maxDistance;
};

#endif
