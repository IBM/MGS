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

#include "TerminalOp.h"
#include "Operation.h"
#include "Predicate.h"
#include "InternalException.h"
#include <memory>
#include <string>
#include <sstream>

void TerminalOp::duplicate(std::auto_ptr<Operation>& rv) const
{
   rv.reset(new TerminalOp(*this));
}

void TerminalOp::operate(Predicate* p1, Predicate* p2,  Predicate* cur) 
{
}

TerminalOp::~TerminalOp() 
{

}


