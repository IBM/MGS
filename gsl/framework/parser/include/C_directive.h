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

#ifndef C_directive_H
#define C_directive_H
#include "Copyright.h"

#include "C_production.h"

class LensContext;
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
      virtual void internalExecute(LensContext *);
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
