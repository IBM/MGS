// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_TRIGGER_H
#define C_TRIGGER_H
#include "Copyright.h"

#include <vector>
#include <string>
#include "C_production.h"

class C_query_path_product;
class C_declarator;
class LensContext;
class DataItem;
class Trigger;
class SyntaxError;

class C_trigger : public C_production
{
   public:
      enum Type {_BASIC, _SINGLE, _AND, _OR, _XOR};
      C_trigger(const C_trigger&);
      C_trigger(C_query_path_product *, SyntaxError *);
      C_trigger(C_declarator *, SyntaxError *);
      C_trigger(C_trigger *, Type, SyntaxError *);
      C_trigger(C_trigger *, C_trigger *, Type, SyntaxError *);
      virtual ~C_trigger();
      virtual C_trigger* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      Trigger* getTrigger() { 
	 return _trigger; 
      }

   private:
      C_query_path_product* _queryPathProduct;
      C_declarator* _declarator;
      Trigger* _trigger;
      C_trigger* _ct1;
      C_trigger* _ct2; 
      Type _type;
};
#endif
