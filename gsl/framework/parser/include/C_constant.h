// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
