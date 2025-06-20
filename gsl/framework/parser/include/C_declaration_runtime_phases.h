// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_DECLARATION_RUNTIME_PHASES_H
#define C_DECLARATION_RUNTIME_PHASES_H
#include "Copyright.h"

#include <vector>

#include "C_declaration.h"

class GslContext;
class DataItem;
class C_phase;
class C_phase_list;

class C_declaration_runtime_phases : public C_declaration
{
   public:
      C_declaration_runtime_phases(const C_declaration_runtime_phases&);
      C_declaration_runtime_phases(C_phase_list *, SyntaxError *);
      virtual ~C_declaration_runtime_phases();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual C_declaration* duplicate() const;

   private:
      C_phase_list* _phaseList;
};
#endif
