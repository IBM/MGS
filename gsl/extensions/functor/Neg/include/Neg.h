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

#ifndef Neg_H
#define Neg_H

#include "Lens.h"
#include "CG_NegBase.h"
#include "LensContext.h"
#include <memory>

class Neg : public CG_NegBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      double userExecute(LensContext* CG_c);
      Neg();
      virtual ~Neg();
      virtual void duplicate(std::unique_ptr<Neg>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_NegBase>&& dup) const;
};

#endif
