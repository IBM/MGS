#include <memory>
// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
      virtual void duplicate(std::unique_ptr<Method>&& dup) const;
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
