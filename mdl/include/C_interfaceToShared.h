// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
   protected:
      using C_interfaceMapping::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_interfaceToShared();
      C_interfaceToShared(const std::string& interface, 
			  const std::string& interfaceMember,
			  C_identifierList* member); 
      virtual void duplicate(std::unique_ptr<C_interfaceToShared>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_interfaceMapping>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_interfaceToShared();
};


#endif // C_interfaceToShared_H
