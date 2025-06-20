// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ConstructorMethod_H
#define ConstructorMethod_H
#include "Mdl.h"

#include "Method.h"
#include <string>
#include <vector>
#include <memory>

class DataType;
class Attribute;

class ConstructorMethod : public Method
{
   public:
      ConstructorMethod();
      ConstructorMethod(const std::string& name, 
			const std::string& returnStr = "",
			const std::string& functionBody = "",
			const std::string& initializationStr = "");
      virtual void duplicate(std::unique_ptr<Method>&& dup) const;
      virtual ~ConstructorMethod();
      const std::string& getInitializationStr() const;
      void setInitializationStr(const std::string& constructorStr);
   protected:
      virtual std::string printConstructorExtra();
      void internalAddConstructorInitializer(
	 const std::vector<Attribute*>& attributes,
	 const std::string& beginning,
	 const std::string& copyFrom = "");
      virtual void callInitMethod(
	 const std::vector<Attribute*>::const_iterator& it,
	 std::string& initStr, const std::string& copyFrom);
      
   private:
      std::string _initializationStr;
};

#endif
