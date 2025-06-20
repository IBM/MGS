// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
      virtual void duplicate(std::unique_ptr<InterfaceToMember>&& rv) const;
      virtual void duplicate(std::unique_ptr<InterfaceMapping>&& rv) const;
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
