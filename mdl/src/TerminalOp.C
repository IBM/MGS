// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
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


