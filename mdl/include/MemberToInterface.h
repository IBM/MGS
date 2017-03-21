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

#ifndef MemberToInterface_H
#define MemberToInterface_H
#include "Mdl.h"

#include <memory>
#include "InterfaceMapping.h"

class Class;

class MemberToInterface : public InterfaceMapping {

   public:
      MemberToInterface(Interface* interface = 0);
      virtual void duplicate(std::auto_ptr<MemberToInterface>& rv) const;
      virtual void duplicate(std::auto_ptr<InterfaceMapping>& rv) const;
      virtual ~MemberToInterface();
      bool checkAllMapped();

      std::string getMemberToInterfaceString(
	 const std::string& interfaceName) const;

      void setupAccessorMethods(Class& instance) const;
      void setupProxyAccessorMethods(Class& instance) const;
      bool hasMemberDataType(const std::string& name) const;

      std::string getServiceNameCode(const std::string& tab) const;

   protected:
      virtual void checkAndExtraWork(const std::string& name,
				     DataType* member, 
				     const DataType* interface, bool amp);
};


#endif // MemberToInterface_H
