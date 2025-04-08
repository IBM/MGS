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

#ifndef NameReturnValue_H
#define NameReturnValue_H

#include "Lens.h"
#include "CG_NameReturnValueBase.h"
#include "LensContext.h"
#include "NDPairList.h"
#include <memory>

class NameReturnValue : public CG_NameReturnValueBase
{
   public:
      void userInitialize(LensContext* CG_c, CustomString& s, Functor*& f);
      std::unique_ptr<NDPairList> userExecute(LensContext* CG_c);
      NameReturnValue();
      virtual ~NameReturnValue();
      virtual void duplicate(std::unique_ptr<NameReturnValue>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_NameReturnValueBase>&& dup) const;
};

#endif
