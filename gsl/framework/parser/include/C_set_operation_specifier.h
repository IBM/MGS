// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_set_operation_specifier_H
#define C_set_operation_specifier_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class C_declarator;
class C_set_operation;
class C_argument_list;
class LensContext;
class SyntaxError;

class C_set_operation_specifier : public C_production
{
   public:
      C_set_operation_specifier(const C_set_operation_specifier&);
      C_set_operation_specifier(C_declarator *, C_set_operation *, 
				std::string, C_declarator *, 
				C_argument_list *, SyntaxError *);
      virtual ~C_set_operation_specifier();
      virtual C_set_operation_specifier* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _dcl;
      C_set_operation* _sop;
      std::string _id;
      C_declarator* _fdl;
      C_argument_list* _arl;
};
#endif
