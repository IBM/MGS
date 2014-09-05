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

#ifndef GSValidOp_H
#define GSValidOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"
#include "InFixOp.h"

class Predicate;

class GSValidOp : public InFixOp {

   public:
      virtual void duplicate(std::auto_ptr<Operation>& rv) const; 
      virtual ~GSValidOp();
      virtual std::string getOp() const;

   protected:
      virtual void check(Predicate* p1, Predicate* p2);

};


#endif // GSValidOp_H
