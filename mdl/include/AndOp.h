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

#ifndef AndOp_H
#define AndOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"
#include "BValidOp.h"

class Predicate;

class AndOp : public BValidOp {

   public:
      virtual void duplicate(std::auto_ptr<Operation>& rv) const;
      virtual ~AndOp();
      virtual std::string getOp() const;
};


#endif // AndOp_H
