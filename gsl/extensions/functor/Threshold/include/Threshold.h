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

#ifndef Threshold_H
#define Threshold_H

#include "Lens.h"
#include "CG_ThresholdBase.h"
#include "LensContext.h"
#include <memory>

class Threshold : public CG_ThresholdBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, double& threshold);
      bool userExecute(LensContext* CG_c);
      Threshold();
      virtual ~Threshold();
      virtual void duplicate(std::auto_ptr<Threshold>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ThresholdBase>& dup) const;
};

#endif
