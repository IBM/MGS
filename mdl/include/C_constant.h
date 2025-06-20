// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_constant_H
#define C_constant_H
#include "Mdl.h"

#include "C_interfaceImplementorBase.h"
#include <memory>

class MdlContext;

class C_constant : public C_interfaceImplementorBase {
   using C_interfaceImplementorBase::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_constant();
      C_constant(const std::string& name, C_interfacePointerList* ipl,
		 C_generalList* gl);
      C_constant(const C_constant& rv);
      virtual void duplicate(std::unique_ptr<C_constant>&& rv) const;
      virtual ~C_constant();
};


#endif // C_constant_H
