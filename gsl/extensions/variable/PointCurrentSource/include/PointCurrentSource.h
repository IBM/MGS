// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2012  All rights reserved
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
      virtual void duplicate(std::auto_ptr<PointCurrentSource>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_PointCurrentSource>& dup) const;
};

#endif
