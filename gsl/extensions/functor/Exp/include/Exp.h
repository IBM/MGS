// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef Exp_H
#define Exp_H

#include "Lens.h"
#include "CG_ExpBase.h"
#include "LensContext.h"
#include <memory>

class Exp : public CG_ExpBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      double userExecute(LensContext* CG_c);
      Exp();
      virtual ~Exp();
      virtual void duplicate(std::auto_ptr<Exp>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ExpBase>& dup) const;
};

#endif
