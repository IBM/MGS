// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-14-2018
//
// (C) Copyright IBM Corp. 2005-2018  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef InterfaceToMember_H
#define InterfaceToMember_H
#include "Mdl.h"

#include <memory>
#include <vector>
#include <set>
#include "InterfaceMapping.h"

class InterfaceToMember : public InterfaceMapping {

   public:
      enum MappingType {ONETOONE, ONETOMANY};
      InterfaceToMember(Interface* interface = 0);
      virtual void duplicate(std::auto_ptr<InterfaceToMember>& rv) const;
      virtual void duplicate(std::auto_ptr<InterfaceMapping>& rv) const;
      virtual ~InterfaceToMember();

      std::string getInterfaceToMemberCode(
	 const std::string& tab, 
	 std::set<std::string>& requiredIncreases) const;

      /* add 'dummy' to support adding code to :addPreNode_Dummy */
      std::string getInterfaceToMemberCode(
	 const std::string& tab, 
	 std::set<std::string>& requiredIncreases, 
	 MachineType mach_type,
	 bool dummy=0,
	 const std::string& className="") const;

      std::string getInterfaceToMemberString(
	 const std::string& interfaceName) const;

   protected:
      virtual void checkAndExtraWork(const std::string& name,
				     DataType* member, 
				     const DataType* interface, bool amp);
   private:
      std::vector<MappingType> _mappingType;

};


#endif // InterfaceToMember_H
