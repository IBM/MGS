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

#ifndef SrcRefDoGWeightModifier_H
#define SrcRefDoGWeightModifier_H
#include "Lens.h"

#include "CG_SrcRefDoGWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class SrcRefDoGWeightModifier : public CG_SrcRefDoGWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma1, float& max1, float& sigma2, float& max2, int& wrapDistance);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcRefDoGWeightModifier();
      virtual ~SrcRefDoGWeightModifier();
      virtual void duplicate(std::unique_ptr<SrcRefDoGWeightModifier>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SrcRefDoGWeightModifierBase>&& dup) const;

      float _sigma1;
      float _max1;
      float _sigma2;
      float _max2;
      int _wrapDistance;
};

#endif
