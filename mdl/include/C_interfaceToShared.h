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

#ifndef C_interfaceToShared_H
#define C_interfaceToShared_H
#include "Mdl.h"

#include "C_interfaceMapping.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_identifierList;

class C_interfaceToShared : public C_interfaceMapping {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_interfaceToShared();
      C_interfaceToShared(const std::string& interface, 
			  const std::string& interfaceMember,
			  C_identifierList* member); 
      virtual void duplicate(std::auto_ptr<C_interfaceToShared>& rv) const;
      virtual void duplicate(std::auto_ptr<C_interfaceMapping>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_interfaceToShared();
};


#endif // C_interfaceToShared_H
