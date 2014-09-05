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

#ifndef NotEqualOp_H
#define NotEqualOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"
#include "AllValidOp.h"

class Predicate;

class NotEqualOp : public AllValidOp {

   public:
      virtual void duplicate(std::auto_ptr<Operation>& rv) const;
      virtual ~NotEqualOp();
      virtual std::string getOp() const;

};


#endif // NotEqualOp_H
