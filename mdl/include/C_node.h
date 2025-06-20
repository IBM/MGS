// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_node_H
#define C_node_H
#include "Mdl.h"

#include "C_sharedCCBase.h"
#include <memory>

class MdlContext;

class C_node : public C_sharedCCBase {
   using C_sharedCCBase::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_node();
      C_node(const std::string& name, C_interfacePointerList* ipl,
	     C_generalList* gl);
      C_node(const C_node& rv);
      virtual void duplicate(std::unique_ptr<C_node>&& rv) const;
      virtual ~C_node();
};


#endif // C_node_H
