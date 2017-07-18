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

#ifndef C_declaration_list_parameter_H
#define C_declaration_list_parameter_H
#include "Copyright.h"

#include <memory>
#include <map>

#include "C_declaration.h"
class C_list_parameter_type;
class C_declarator;
class C_type_specifier;
class C_argument_list;
class LensContext;
class SyntaxError;

class C_declaration_list_parameter : public C_declaration
{
   public:
      C_declaration_list_parameter(const C_declaration_list_parameter&);
      C_declaration_list_parameter(C_type_specifier *, C_declarator *, 
				   C_argument_list *, SyntaxError *);
      C_declaration_list_parameter(C_declarator *, C_argument_list *, 
				   SyntaxError *);
      virtual C_declaration_list_parameter* duplicate() const;
      virtual ~C_declaration_list_parameter();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_type_specifier* _typeSpec;
      C_declarator* _declarator;
      C_argument_list* _argumentList;

};
#endif
