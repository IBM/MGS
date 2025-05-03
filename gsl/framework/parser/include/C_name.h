// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_name_H
#define C_name_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class C_name;
class GslContext;
class SyntaxError;

class C_name : public C_production
{
   public:
      C_name(const C_name&);
      C_name(std::string *, SyntaxError *);
      virtual ~C_name();
      virtual C_name* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::string getName() {
	 return *_name;
      }

   private:
      std::string *_name;
};
#endif
