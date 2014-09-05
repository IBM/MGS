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

#ifndef ReverseFunctor_H
#define ReverseFunctor_H
#include "Lens.h"

#include "CG_ReverseFunctorBase.h"
#include "LensContext.h"
#include <memory>

class ReverseFunctor : public CG_ReverseFunctorBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      void userExecute(LensContext* CG_c);
      ReverseFunctor();
      virtual ~ReverseFunctor();
      virtual void duplicate(std::auto_ptr<ReverseFunctor>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ReverseFunctorBase>& dup) const;
};
#endif
