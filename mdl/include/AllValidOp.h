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

#ifndef AllValidOp_H
#define AllValidOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"
#include "InFixOp.h"

class Predicate;

class AllValidOp : public InFixOp {

   public:
      virtual void duplicate(std::auto_ptr<Operation>& rv) const;
      virtual ~AllValidOp();
      virtual std::string getOp() const;

   protected:
      virtual void check(Predicate* p1, Predicate* p2);

};


#endif // AllValidOp_H
