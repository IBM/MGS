// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_edge_H
#define C_edge_H
#include "Mdl.h"

#include "C_sharedCCBase.h"
#include <memory>

class MdlContext;

class C_edge : public C_sharedCCBase {
   using C_sharedCCBase::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_edge();
      C_edge(const std::string& name, C_interfacePointerList* ipl
			 , C_generalList* gl);
      C_edge(const C_edge& rv);
      virtual void duplicate(std::unique_ptr<C_edge>&& rv) const;
      virtual ~C_edge();
};


#endif // C_edge_H
