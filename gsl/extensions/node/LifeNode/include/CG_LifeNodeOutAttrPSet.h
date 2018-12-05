// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CG_LifeNodeOutAttrPSet_H
#define CG_LifeNodeOutAttrPSet_H

#include "Lens.h"
#include "NDPair.h"
#include "NDPairList.h"
#include "ParameterSet.h"
#include "SyntaxErrorException.h"
#include <memory>
#include <typeinfo>

class CG_LifeNodeOutAttrPSet : public ParameterSet
{
   public:
      virtual void set(NDPairList& ndplist);
      CG_LifeNodeOutAttrPSet();
      virtual ~CG_LifeNodeOutAttrPSet();
      virtual void duplicate(std::unique_ptr<CG_LifeNodeOutAttrPSet>& dup) const;
      virtual void duplicate(std::unique_ptr<ParameterSet>& dup) const;
};

#endif
