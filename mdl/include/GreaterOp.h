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

#ifndef GreaterOp_H
#define GreaterOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"
#include "GSValidOp.h"

class Predicate;

class GreaterOp : public GSValidOp {

   public:
      virtual void duplicate(std::auto_ptr<Operation>& rv) const;
      virtual ~GreaterOp();
      virtual std::string getOp() const;

};


#endif // GreaterOp_H
