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

#ifndef Round_H
#define Round_H

#include "Lens.h"
#include "CG_RoundBase.h"
#include "LensContext.h"
#include <memory>

class Round : public CG_RoundBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      double userExecute(LensContext* CG_c);
      Round();
      virtual ~Round();
      virtual void duplicate(std::auto_ptr<Round>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_RoundBase>& dup) const;
};

#endif
