// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_psetToInstance_H
#define C_psetToInstance_H
#include "Mdl.h"

#include "C_psetMapping.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;

class C_psetToInstance : public C_psetMapping {

   public:  
      using C_psetMapping::duplicate;
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_psetToInstance();
      C_psetToInstance(const std::string& psetMember,
		       C_identifierList* member); 
      virtual void duplicate(std::unique_ptr<C_psetToInstance>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_psetMapping>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_psetToInstance();
};


#endif // C_psetToInstance_H
