// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_variable_H
#define C_variable_H
#include "Mdl.h"

#include "C_connectionCCBase.h"
#include <memory>

class MdlContext;

class C_variable : public C_connectionCCBase {
   using C_connectionCCBase::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_variable();
      C_variable(const std::string& name, C_interfacePointerList* ipl
			 , C_generalList* gl);
      C_variable(const C_variable& rv);
      virtual void duplicate(std::unique_ptr<C_variable>&& rv) const;
      virtual ~C_variable();
};


#endif // C_variable_H
