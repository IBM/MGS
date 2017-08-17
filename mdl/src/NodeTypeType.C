// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "NodeTypeType.h"
#include "LensType.h"
#include "DataType.h"
#include <string>
#include <memory>

void NodeTypeType::duplicate(std::auto_ptr<DataType>& rv) const
{
   rv.reset(new NodeTypeType(*this));
}

std::string NodeTypeType::getDescriptor() const
{
   return "NodeType";
}

NodeTypeType::~NodeTypeType() 
{
}

