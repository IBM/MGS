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

#ifndef Scale_H
#define Scale_H

#include "Lens.h"
#include "CG_ScaleBase.h"
#include "LensContext.h"
#include <memory>

class Scale : public CG_ScaleBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, double& scale);
      double userExecute(LensContext* CG_c);
      Scale();
      virtual ~Scale();
      virtual void duplicate(std::unique_ptr<Scale>& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ScaleBase>& dup) const;
};

#endif
