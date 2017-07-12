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

#ifndef InFixOp_H
#define InFixOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"

class Predicate;

class InFixOp : public Operation {

   public:
      virtual void duplicate(std::auto_ptr<Operation>& rv) const;
      virtual ~InFixOp();
      virtual void operate(Predicate* p1, Predicate* p2,  Predicate* cur);
      virtual std::string getOp() const;

   protected:
      virtual void check(Predicate* p1, Predicate* p2);

};


#endif // InFixOp_H
