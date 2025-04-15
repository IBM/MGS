// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_constant_H
#define C_constant_H
#include "Copyright.h"

#include "C_production.h"

class LensContext;
class SyntaxError;

class C_constant : public C_production
{
   public:
      enum Type {_INT, _FLOAT};
      C_constant(const C_constant&);
      C_constant(int, SyntaxError *);
      C_constant(double, SyntaxError *) ;
      virtual C_constant* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual ~C_constant ();
      Type getType() {
	 return _type; 
      }
      int getInt();
      float getFloat();

   private:
      Type _type;
      int _intValue;
      double _floatValue;
};
#endif
