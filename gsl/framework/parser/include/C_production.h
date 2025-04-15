// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_production_H
#define C_production_H
#include "Copyright.h"

#include <string>

class LensContext;
class SyntaxError;

class C_production
{
   public:
      C_production(SyntaxError* error);
      C_production(const C_production&);
      virtual ~C_production ();
      virtual C_production* duplicate() const = 0;
      bool isError();
      void setError();
      void printErrorMessage();
      virtual void throwError(const std::string&);
      virtual void execute(LensContext *);
      virtual void checkChildren() {};
      virtual void recursivePrint() {};
   protected:
      virtual void internalExecute(LensContext *) = 0;
      SyntaxError *_error;
};
#endif
