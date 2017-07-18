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

#ifndef ParanthesisOp_H
#define ParanthesisOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"

class Predicate;

class ParanthesisOp : public Operation {

   public:
      virtual void duplicate(std::auto_ptr<Operation>& rv) const;
      virtual ~ParanthesisOp();
      virtual void operate(Predicate* p1, Predicate* p2, Predicate* cur);
};


#endif // ParanthesisOp_H
