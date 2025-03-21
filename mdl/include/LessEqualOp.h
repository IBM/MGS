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

#ifndef LessEqualOp_H
#define LessEqualOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"
#include "GSValidOp.h"

class Predicate;

class LessEqualOp : public GSValidOp {

   public:
      virtual void duplicate(std::unique_ptr<Operation>&& rv) const;
      virtual ~LessEqualOp();
      virtual std::string getOp() const;

};


#endif // LessEqualOp_H
