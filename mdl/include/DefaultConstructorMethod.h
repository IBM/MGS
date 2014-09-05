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

#ifndef DefaultConstructorMethod_H
#define DefaultConstructorMethod_H
#include "Mdl.h"

#include "ConstructorMethod.h"
#include <string>
#include <vector>

class Attribute;

class DefaultConstructorMethod : public ConstructorMethod
{
   public:
      DefaultConstructorMethod();
      DefaultConstructorMethod(
	 const std::string& name, const std::string& returnStr = "",
	 const std::string& functionBody = "", 
	 const std::string& initializationStr = "");
      virtual void duplicate(std::auto_ptr<Method>& dup) const;
      virtual ~DefaultConstructorMethod();      
      void addDefaultConstructorParameters(
	 const std::vector<Attribute*>& attributes,
	 const std::string& className = "");
      void addDefaultConstructorInitializers(
	 const std::vector<Attribute*>& attributes,
	 const std::string& beginning);
   protected:
      virtual void callInitMethod(
	 const std::vector<Attribute*>::const_iterator& it,
	 std::string& initStr, const std::string& copyFrom);
};

#endif
