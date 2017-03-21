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
      virtual void duplicate(std::auto_ptr<Method>& dup) const;
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
