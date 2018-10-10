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

#ifndef RefAngleModifier_H
#define RefAngleModifier_H
#include "Lens.h"

#include "CG_RefAngleModifierBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class RefAngleModifier : public CG_RefAngleModifierBase
{
   public:
      void userInitialize(LensContext* CG_c, int& directionFlag, int& wrapFlag, Functor*& f);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      RefAngleModifier();
      virtual ~RefAngleModifier();
      virtual void duplicate(std::unique_ptr<RefAngleModifier>& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_RefAngleModifierBase>& dup) const;
};

#endif
