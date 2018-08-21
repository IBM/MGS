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

#ifndef C_interfaceMapping_H
#define C_interfaceMapping_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_identifierList;

class C_interfaceMapping : public C_general {

   public:
      virtual void execute(MdlContext* context);
      C_interfaceMapping();
      C_interfaceMapping(const std::string& interface, 
			 const std::string& interfaceMember,
			 C_identifierList* member,
			 bool amp = false); 

      C_interfaceMapping(const C_interfaceMapping& rv);
      C_interfaceMapping& operator=(const C_interfaceMapping& rv);

      virtual void duplicate(std::auto_ptr<C_interfaceMapping>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_interfaceMapping();

      const std::string& getInterface() const {
	 return _interface;
      }

      const std::string& getInterfaceMember() const {
	 return _interfaceMember;
      }

      const std::string& getMember() const;
      
      bool getSubAttributePathExists() const;

      std::vector<std::string> getSubAttributePath() const;

      bool getAmpersand() const {
	 return _ampersand;
      }

   protected:
      void destructOwnedHeap();
      void copyOwnedHeap(const C_interfaceMapping& rv);

      std::string _interface;
      std::string _interfaceMember;
      C_identifierList* _member;
      bool _ampersand;
};


#endif // C_interfaceMapping_H
