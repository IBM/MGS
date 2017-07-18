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

#include "NodeSetType.h"
#include "LensType.h"
#include "DataType.h"
#include <string>
#include <memory>

void NodeSetType::duplicate(std::auto_ptr<DataType>& rv) const
{
   rv.reset(new NodeSetType(*this));
}

std::string NodeSetType::getDescriptor() const
{
   return "NodeSet";
}

bool NodeSetType::shouldBeOwned() const
{
   return true;
}

NodeSetType::~NodeSetType() 
{
}

