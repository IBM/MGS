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

#ifndef CG_LifeNodeSharedMembers_H
#define CG_LifeNodeSharedMembers_H

#include "Lens.h"
#include "IntDataItem.h"
#include <memory>
#include <sstream>

class NDPairList;

class CG_LifeNodeSharedMembers
{
   public:
      virtual void setUp(const NDPairList& ndplist);
      CG_LifeNodeSharedMembers();
      virtual ~CG_LifeNodeSharedMembers();
      virtual void duplicate(std::unique_ptr<CG_LifeNodeSharedMembers>& dup) const;
      int tooCrowded;
      int tooSparse;
};

#endif
