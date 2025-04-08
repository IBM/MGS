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

#ifndef ModifyParameterSet_H
#define ModifyParameterSet_H

#include "Lens.h"
#include "CG_ModifyParameterSetBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class ModifyParameterSet : public CG_ModifyParameterSetBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f1, Functor*& f2);
      std::unique_ptr<ParameterSet> userExecute(LensContext* CG_c);
      ModifyParameterSet();
      virtual ~ModifyParameterSet();
      virtual void duplicate(std::unique_ptr<ModifyParameterSet>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ModifyParameterSetBase>&& dup) const;
};

#endif
