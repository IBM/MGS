// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_sharedMapping_H
#define C_sharedMapping_H
#include "Mdl.h"

#include "C_interfaceMapping.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_identifierList;

class C_sharedMapping : public C_interfaceMapping {
   protected:
      using C_interfaceMapping::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_sharedMapping();
      C_sharedMapping(const std::string& interface, 
		      const std::string& interfaceMember, 
		      C_identifierList* dataType,
		      bool amp = false); 
      virtual void duplicate(std::unique_ptr<C_sharedMapping>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_interfaceMapping>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_sharedMapping();
};


#endif // C_sharedMapping_H
