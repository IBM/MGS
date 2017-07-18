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

#ifndef SrcRefPeakedWeightModifier_H
#define SrcRefPeakedWeightModifier_H
#include "Lens.h"

#include "CG_SrcRefPeakedWeightModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class SrcRefPeakedWeightModifier : public CG_SrcRefPeakedWeightModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, float& sigma, float& max, int& wrapDistance);
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      SrcRefPeakedWeightModifier();
      virtual ~SrcRefPeakedWeightModifier();
      virtual void duplicate(std::auto_ptr<SrcRefPeakedWeightModifier>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_SrcRefPeakedWeightModifierBase>& dup) const;

      float _sigma;
      float _max;
      int _wrapDistance;
};

#endif
