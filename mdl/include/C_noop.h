// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_noop_H
#define C_noop_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>

class MdlContext;
class C_typeClassifier;
class C_generalList;

class C_noop : public C_general {
   protected:
      using C_general::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_noop();
      C_noop(const C_noop& rv);
      virtual void duplicate(std::unique_ptr<C_noop>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_noop();
      
};


#endif // C_noop_H
