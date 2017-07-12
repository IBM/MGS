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

#ifndef LessOp_H
#define LessOp_H
#include "Mdl.h"

#include <memory>
#include "Operation.h"
#include "GSValidOp.h"

class Predicate;

class LessOp : public GSValidOp {

   public:
      virtual void duplicate(std::auto_ptr<Operation>& rv) const;
      virtual ~LessOp();
      virtual std::string getOp() const;

};


#endif // LessOp_H
