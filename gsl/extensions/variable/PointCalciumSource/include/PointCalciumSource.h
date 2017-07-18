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

#ifndef PointCalciumSource_H
#define PointCalciumSource_H

#include "Lens.h"
#include "CG_PointCalciumSource.h"
#include <memory>

class PointCalciumSource : public CG_PointCalciumSource
{
   public:
      void stimulate(RNG& rng);
      virtual void setCaCurrent(Trigger* trigger, NDPairList* ndPairList);
      PointCalciumSource();
      virtual ~PointCalciumSource();
      virtual void duplicate(std::auto_ptr<PointCalciumSource>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_PointCalciumSource>& dup) const;
};

#endif
