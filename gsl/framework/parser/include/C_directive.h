// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_directive_H
#define C_directive_H
#include "Copyright.h"

#include "C_production.h"

class GslContext;
class C_functor_specifier;
class C_trigger_specifier;
class C_system_call;
class DataItem;
class SyntaxError;

class C_directive : public C_production
{
   public:
      C_directive(const C_directive&);
      C_directive(C_functor_specifier *, SyntaxError * error);
      C_directive(C_trigger_specifier*, SyntaxError * error);
      C_directive(C_system_call*, SyntaxError * error);
      virtual ~C_directive();
      virtual C_directive* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // Will return null if void
      const DataItem* getRVal() const;

   private:
      C_functor_specifier* _functorSpec;
      C_trigger_specifier* _triggerSpec;
      C_system_call* _systemCall;
};
#endif
