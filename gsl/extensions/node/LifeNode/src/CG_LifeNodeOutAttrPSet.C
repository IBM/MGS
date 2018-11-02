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
#include "CG_LifeNodeOutAttrPSet.h"
#include <sstream>
#include "NDPair.h"
#include "NDPairList.h"
#include "ParameterSet.h"
#include "SyntaxErrorException.h"
#include <memory>
#include <typeinfo>

void CG_LifeNodeOutAttrPSet::set(NDPairList& ndplist) 
{
}

CG_LifeNodeOutAttrPSet::CG_LifeNodeOutAttrPSet() 
   : ParameterSet()
{
}

CG_LifeNodeOutAttrPSet::~CG_LifeNodeOutAttrPSet() 
{
}

void CG_LifeNodeOutAttrPSet::duplicate(std::unique_ptr<CG_LifeNodeOutAttrPSet>& dup) const
{
   dup.reset(new CG_LifeNodeOutAttrPSet(*this));
}

void CG_LifeNodeOutAttrPSet::duplicate(std::unique_ptr<ParameterSet>& dup) const
{
   dup.reset(new CG_LifeNodeOutAttrPSet(*this));
}

