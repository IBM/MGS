// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_system_call_H
#define C_system_call_H
#include "Copyright.h"

#include <string>

#include "C_production.h"

class GslContext;
class SyntaxError;

class C_system_call : public C_production
{
   public:
      C_system_call(std::string *, SyntaxError *);
      C_system_call(const C_system_call&);
      virtual ~C_system_call();
      virtual C_system_call* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      std::string* _command;
};
#endif
