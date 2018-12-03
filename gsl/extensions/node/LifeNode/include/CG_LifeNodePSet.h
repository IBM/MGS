// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-12-03-2018
//
//  (C) Copyright IBM Corp. 2005-2018  All rights reserved   .
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CG_LifeNodePSet_H
#define CG_LifeNodePSet_H

#include "Lens.h"
#include "DataItem.h"
#include "DataItemArrayDataItem.h"
#include "IntArrayDataItem.h"
#include "IntDataItem.h"
#include "NDPair.h"
#include "NDPairList.h"
#include "ParameterSet.h"
#include "ShallowArray.h"
#include "SyntaxErrorException.h"
#include <memory>
#include <typeinfo>

class CG_LifeNodePSet : public ParameterSet
{
   public:
      virtual void set(NDPairList& ndplist);
      CG_LifeNodePSet();
      virtual ~CG_LifeNodePSet();
      virtual void duplicate(std::unique_ptr<CG_LifeNodePSet>& dup) const;
      virtual void duplicate(std::unique_ptr<ParameterSet>& dup) const;
      int value;
      int publicValue;
      ShallowArray< int* > neighbors;
};

#endif
