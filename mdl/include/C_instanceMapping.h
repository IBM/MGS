// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_instanceMapping_H
#define C_instanceMapping_H
#include "Mdl.h"

#include "C_interfaceMapping.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_identifierList;

class C_instanceMapping : public C_interfaceMapping {

   public:
      using C_interfaceMapping::duplicate;
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_instanceMapping();
      C_instanceMapping(const std::string& interface, 
			const std::string& interfaceMember,
			C_identifierList* dataType,
			bool amp = false); 
      
      virtual void duplicate(std::unique_ptr<C_instanceMapping>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_interfaceMapping>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_instanceMapping();
};


#endif // C_instanceMapping_H
