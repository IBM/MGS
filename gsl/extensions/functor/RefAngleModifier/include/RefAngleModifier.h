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
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      RefAngleModifier();
      virtual ~RefAngleModifier();
      virtual void duplicate(std::auto_ptr<RefAngleModifier>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_RefAngleModifierBase>& dup) const;
};

#endif
