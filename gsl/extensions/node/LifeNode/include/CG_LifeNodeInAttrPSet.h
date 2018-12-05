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

#ifndef CG_LifeNodeInAttrPSet_H
#define CG_LifeNodeInAttrPSet_H

#include "Lens.h"
#include "NDPair.h"
#include "NDPairList.h"
#include "ParameterSet.h"
#include "SyntaxErrorException.h"
#include <memory>
#include <typeinfo>

class CG_LifeNodeInAttrPSet : public ParameterSet
{
   public:
      virtual void set(NDPairList& ndplist);
      CG_LifeNodeInAttrPSet();
      virtual ~CG_LifeNodeInAttrPSet();
      virtual void duplicate(std::unique_ptr<CG_LifeNodeInAttrPSet>& dup) const;
      virtual void duplicate(std::unique_ptr<ParameterSet>& dup) const;
};

#endif
