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

#ifndef PhaseTypeShared_H
#define PhaseTypeShared_H
#include "Mdl.h"

#include "PhaseType.h"
#include <memory>
#include <string>

class PhaseTypeShared : public PhaseType {

   public:
      PhaseTypeShared();
      virtual void duplicate(std::auto_ptr<PhaseType>& rv) const;
      virtual ~PhaseTypeShared();
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


#endif // PhaseTypeShared _H
