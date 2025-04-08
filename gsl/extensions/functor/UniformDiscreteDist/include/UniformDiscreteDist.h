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

#ifndef UniformDiscreteDist_H
#define UniformDiscreteDist_H

#include "Lens.h"
#include "CG_UniformDiscreteDistBase.h"
#include "LensContext.h"
#include <memory>

class UniformDiscreteDist : public CG_UniformDiscreteDistBase
{
   public:
      void userInitialize(LensContext* CG_c, double& n1, double& n2);
      int userExecute(LensContext* CG_c);
      UniformDiscreteDist();
      virtual ~UniformDiscreteDist();
      virtual void duplicate(std::unique_ptr<UniformDiscreteDist>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_UniformDiscreteDistBase>&& dup) const;
};

#endif
