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

#include "Lens.h"
#include "CG_LifeNodeInAttrPSet.h"
#include <sstream>
#include "NDPair.h"
#include "NDPairList.h"
#include "ParameterSet.h"
#include "SyntaxErrorException.h"
#include <memory>
#include <typeinfo>

void CG_LifeNodeInAttrPSet::set(NDPairList& ndplist) 
{
}

CG_LifeNodeInAttrPSet::CG_LifeNodeInAttrPSet() 
   : ParameterSet()
{
}

CG_LifeNodeInAttrPSet::~CG_LifeNodeInAttrPSet() 
{
}

void CG_LifeNodeInAttrPSet::duplicate(std::unique_ptr<CG_LifeNodeInAttrPSet>& dup) const
{
   dup.reset(new CG_LifeNodeInAttrPSet(*this));
}

void CG_LifeNodeInAttrPSet::duplicate(std::unique_ptr<ParameterSet>& dup) const
{
   dup.reset(new CG_LifeNodeInAttrPSet(*this));
}

