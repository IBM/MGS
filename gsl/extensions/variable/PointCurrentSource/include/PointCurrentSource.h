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

#ifndef PointCurrentSource_H
#define PointCurrentSource_H

#include "Lens.h"
#include "CG_PointCurrentSource.h"
#include <memory>

class PointCurrentSource : public CG_PointCurrentSource
{
   public:
      virtual void stimulate(RNG& rng);
      virtual void setCurrent(Trigger* trigger, NDPairList* ndPairList);
      PointCurrentSource();
      virtual ~PointCurrentSource();
      virtual void duplicate(std::unique_ptr<PointCurrentSource>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_PointCurrentSource>& dup) const;
};

#endif
