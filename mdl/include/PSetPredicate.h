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

#ifndef PSetPredicate_H
#define PSetPredicate_H
#include "Mdl.h"

#include "Predicate.h"
#include <memory>
#include <string>

class StructType;
class Operation;

class PSetPredicate : public Predicate{

   public:
      PSetPredicate();
      PSetPredicate(Operation* op, const std::string& name);
      virtual void duplicate(std::auto_ptr<Predicate>& rv) const;
      virtual ~PSetPredicate();
      virtual void setPSet(StructType& type);
      
   private:
      std::string getTypeUsingPSet(StructType& type);
};


#endif // PSetPredicate_H
