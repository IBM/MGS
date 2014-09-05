// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
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
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcRefSumRsqrdInvWeightModifier();
      virtual ~SrcRefSumRsqrdInvWeightModifier();
      virtual void duplicate(std::auto_ptr<SrcRefSumRsqrdInvWeightModifier>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_SrcRefSumRsqrdInvWeightModifierBase>& dup) const;

      bool _setDistance;
      std::map<float, float> _radiusMap;
      int _maxDistance;
};

#endif
