// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PhaseType_H
#define PhaseType_H
#include "Mdl.h"
#include "Constants.h"

#include <memory>
#include <string>

class Class;
class Method;

class PhaseType {

   public:
      PhaseType();
      virtual void duplicate(std::unique_ptr<PhaseType>&& rv) const = 0;
      virtual ~PhaseType();

      virtual std::string getType() const =0;
      virtual std::string getParameter(
	 const std::string& componentType) const =0;
      virtual void generateInstancePhaseMethod(
	 Class& c, const std::string& name, const std::string& instanceType, 
	 const std::string& componentType, const std::string& workUnitName) const = 0;
      std::string getInstancePhaseMethodName(const std::string& name,
					     const std::string &workUnitName,
					     MachineType mach_type = MachineType::CPU) const;

      virtual std::string getWorkUnitsMethodBody(
	 const std::string& tab, const std::string& workUnits,
	 const std::string& instanceType, const std::string& name, 
	 const std::string& componentType) const=0;

   protected:
      void getInternalInstancePhaseMethod(
	 std::unique_ptr<Method>&& method, const std::string& name, 
	 const std::string& componentType,
	 const std::string& workUnitName,
	 MachineType mach_type = MachineType::CPU) const;
};


#endif // PhaseType _H
