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

#ifndef SharedPredicate_H
#define SharedPredicate_H
#include "Mdl.h"

#include "Predicate.h"
#include <memory>
#include <string>

class StructType;
class Operation;

class SharedPredicate : public Predicate{

   public:
      SharedPredicate();
      SharedPredicate(Operation* op, const std::string& name);
      virtual void duplicate(std::auto_ptr<Predicate>& rv) const;
      virtual ~SharedPredicate();
      virtual void setShareds(const MemberContainer<DataType>& shareds);
      virtual bool checkShareds();
      
   private:
      std::string getTypeUsingShared(const MemberContainer<DataType>& shareds);
};


#endif // SharedPredicate_H
