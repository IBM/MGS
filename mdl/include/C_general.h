// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_general_H
#define C_general_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;

class C_general : public C_production {
   protected:
      using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_general();
      C_general(const C_general& rv);
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_general();

};


#endif // C_general_H
