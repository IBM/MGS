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

#ifndef PhaseType_H
#define PhaseType_H
#include "Mdl.h"

#include <memory>
#include <string>

class Class;
class Method;

class PhaseType {

   public:
      PhaseType();
      virtual void duplicate(std::auto_ptr<PhaseType>& rv) const = 0;
      virtual ~PhaseType();

      virtual std::string getType() const =0;
      virtual std::string getParameter(
	 const std::string& componentType) const =0;
      virtual void generateInstancePhaseMethod(
	 Class& c, const std::string& name, const std::string& instanceType, 
	 const std::string& componentType) const = 0;
      std::string getInstancePhaseMethodName(const std::string& name) const;

      virtual std::string getWorkUnitsMethodBody(
	 const std::string& tab, const std::string& workUnits,
	 const std::string& instanceType, const std::string& name, 
	 const std::string& componentType) const=0;

   protected:
      void getInternalInstancePhaseMethod(
	 std::auto_ptr<Method>& method, const std::string& name, 
	 const std::string& componentType) const;
};


#endif // PhaseType _H
