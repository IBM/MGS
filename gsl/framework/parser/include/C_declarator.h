// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declarator_H
#define C_declarator_H
#include "Copyright.h"

#include <string>

#include "C_production.h"

class GslContext;
class SyntaxError;

class C_declarator : public C_production
{
   public:
      C_declarator(std::string *, SyntaxError *);
      C_declarator(const C_declarator&);
      virtual ~C_declarator();
      virtual C_declarator* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessor method
      const std::string& getName();

   private:
      std::string* _name;
};
#endif
