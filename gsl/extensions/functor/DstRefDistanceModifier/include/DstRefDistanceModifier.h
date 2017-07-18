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

#ifndef DstRefDistanceModifier_H
#define DstRefDistanceModifier_H
#include "Lens.h"

#include "CG_DstRefDistanceModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class DstRefDistanceModifier : public CG_DstRefDistanceModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      DstRefDistanceModifier();
      virtual ~DstRefDistanceModifier();
      virtual void duplicate(std::auto_ptr<DstRefDistanceModifier>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_DstRefDistanceModifierBase>& dup) const;
};

#endif
