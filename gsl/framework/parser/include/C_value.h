// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_value_H
#define C_value_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_value : public C_production
{
   public:
      C_value(const C_value& rv);
      C_value(std::string *, SyntaxError *);
      virtual ~C_value();
      virtual C_value* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::string getValue() {
	 return *_value;
      }

   private:
      std::string* _value;
};
#endif
