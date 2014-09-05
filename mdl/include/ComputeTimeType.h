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
      virtual void duplicate(std::auto_ptr<ComputeTimeType>& rv) const = 0;
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
	 std::auto_ptr<Method>& method, const std::string& name, 
	 const std::string& componentType) const;
};


#endif // ComputeTimeType _H
