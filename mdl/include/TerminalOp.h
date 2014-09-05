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

#ifndef TerminalOp_H
#define TerminalOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"

class Predicate;

class TerminalOp : public Operation {

   public:
      virtual void duplicate(std::auto_ptr<Operation>& rv) const;
      virtual ~TerminalOp();
      virtual void operate(Predicate* p1, Predicate* p2,  Predicate* cur);
};


#endif // TerminalOp_H
