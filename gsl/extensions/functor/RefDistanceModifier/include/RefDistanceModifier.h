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

#ifndef RefDistanceModifier_H
#define RefDistanceModifier_H
#include "Lens.h"

#include "CG_RefDistanceModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class RefDistanceModifier : public CG_RefDistanceModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, int& directionFlag, int& WrapFlag, Functor*& f);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      RefDistanceModifier();
      virtual ~RefDistanceModifier();
      virtual void duplicate(std::unique_ptr<RefDistanceModifier>& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_RefDistanceModifierBase>& dup) const;
};

#endif
