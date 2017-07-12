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

#ifndef GreaterEqualOp_H
#define GreaterEqualOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"
#include "GSValidOp.h"

class Predicate;

class GreaterEqualOp : public GSValidOp {

   public:
      virtual void duplicate(std::auto_ptr<Operation>& rv) const;
      virtual ~GreaterEqualOp();
      virtual std::string getOp() const;

};


#endif // GreaterEqualOp_H
