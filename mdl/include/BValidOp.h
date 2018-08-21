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

#ifndef BValidOp_H
#define BValidOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"
#include "InFixOp.h"

class Predicate;

class BValidOp : public InFixOp {

   public:
      virtual void duplicate(std::auto_ptr<Operation>& rv) const;
      virtual ~BValidOp();
      virtual std::string getOp() const;

   protected:
      virtual void check(Predicate* p1, Predicate* p2);

};


#endif // BValidOp_H
