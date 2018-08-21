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

#ifndef C_TRIGGER_SPECIFIER_H
#define C_TRIGGER_SPECIFIER_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class C_declarator;
class LensContext;
class C_trigger;
class SyntaxError;
class C_ndpair_clause_list;

class C_trigger_specifier : public C_production
{
   public:
      C_trigger_specifier(const C_trigger_specifier&);
      C_trigger_specifier(C_declarator *, C_declarator *, 
			  C_ndpair_clause_list *, C_trigger *, SyntaxError *);
      C_trigger_specifier(std::string, C_trigger *, SyntaxError *);
      virtual ~C_trigger_specifier();
      virtual C_trigger_specifier* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _triggerable;
      C_declarator* _action;
      C_trigger* _trigger;
      C_ndpair_clause_list* _ndpairList;
      std::string _triggerableSpecifier;
};
#endif
