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

#ifndef LessEqualOp_H
#define LessEqualOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"
#include "GSValidOp.h"

class Predicate;

class LessEqualOp : public GSValidOp {

   public:
      virtual void duplicate(std::auto_ptr<Operation>& rv) const;
      virtual ~LessEqualOp();
      virtual std::string getOp() const;

};


#endif // LessEqualOp_H
