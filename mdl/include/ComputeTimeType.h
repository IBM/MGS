// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ComputeTimeType_H
#define ComputeTimeType_H
#include "Mdl.h"

#include <memory>
#include <string>

class Class;
class Method;

class ComputeTimeType {

   public:
      ComputeTimeType();
      virtual void duplicate(std::unique_ptr<ComputeTimeType>&& rv) const = 0;
      virtual ~ComputeTimeType();

      virtual std::string getType() const =0;
      virtual std::string getParameter(
	 const std::string& componentType) const =0;
      virtual void generateInstanceComputeTimeMethod(
	 Class& c, const std::string& name, const std::string& instanceType, 
	 const std::string& componentType) const = 0;
      std::string getInstanceComputeTimeMethodName(const std::string& name) const;

      virtual std::string getWorkUnitsMethodBody(
	 const std::string& tab, const std::string& workUnits,
	 const std::string& instanceType, const std::string& name, 
	 const std::string& componentType) const=0;

   protected:
      void getInternalInstanceComputeTimeMethod(
	 std::unique_ptr<Method>&& method, const std::string& name, 
	 const std::string& componentType) const;
};


#endif // ComputeTimeType _H
