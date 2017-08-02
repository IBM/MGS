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

#ifndef InstancePredicate_H
#define InstancePredicate_H
#include "Mdl.h"

#include "Predicate.h"
#include <memory>
#include <string>

class StructType;
class Operation;

class InstancePredicate : public Predicate{

   public:
      InstancePredicate();
      InstancePredicate(Operation* op, const std::string& name);
      virtual void duplicate(std::auto_ptr<Predicate>& rv) const;
      virtual ~InstancePredicate();
      virtual void setInstances(const MemberContainer<DataType>& instances);
      
   private:
      std::string getTypeUsingInstance(const MemberContainer<DataType>& instances);
};


#endif // InstancePredicate_H
