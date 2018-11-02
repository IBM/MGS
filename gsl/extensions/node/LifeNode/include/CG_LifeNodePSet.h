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
#if defined(HAVE_GPU) && defined(__NVCC__)
      //TUAN TODO: we may not need to have 'reference' elements array here
      //  otherwise, consider proper allocation
      ShallowArray_Flat< int*, Array_Flat<int>::MemLocation::UNIFIED_MEM > neighbors;
#else
      ShallowArray< int* > neighbors;
#endif
};

#endif
