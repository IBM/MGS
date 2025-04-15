// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PhaseTypeGridLayers_H
#define PhaseTypeGridLayers_H
#include "Mdl.h"

#include "PhaseType.h"
#include <memory>
#include <string>

// This class is not used, that is why it is not in the UML document.
// The functionality is ok though, so I didn't remove it from the 
// project. --sgc

class PhaseTypeGridLayers : public PhaseType {

   public:
      PhaseTypeGridLayers();
      virtual void duplicate(std::unique_ptr<PhaseType>&& rv) const;
      virtual ~PhaseTypeGridLayers();
      virtual std::string getType() const;
      virtual std::string getParameter(const std::string& componentType) const;
      virtual void generateInstancePhaseMethod(
	 Class& c, const std::string& name, const std::string& instanceType, 
	 const std::string& componentType, const std::string& workUnitName) const;    
      virtual std::string getWorkUnitsMethodBody(
	 const std::string& tab, const std::string& workUnits,
	 const std::string& instanceType, const std::string& name, 
	 const std::string& componentType) const;
};


#endif // PhaseTypeGridLayers _H
