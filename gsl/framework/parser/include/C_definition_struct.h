// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_DEFINITION_STRUCT_H
#define C_DEFINITION_STRUCT_H
#include "Copyright.h"

#include <string>
#include <list>

#include "C_definition.h"

class C_declarator;
class C_parameter_type;
class C_parameter_type_list;
class LensContext;
class InstanceFactory;
class SyntaxError;

class C_definition_struct : public C_definition
{
   public:
      C_definition_struct(const C_definition_struct&);
      C_definition_struct(C_declarator *, SyntaxError *);
      virtual ~C_definition_struct();
      virtual C_definition_struct* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessor methods
      std::string getDeclarator();
      InstanceFactory* getInstanceFactory();

   private:
      C_declarator* _declarator;
      InstanceFactory* _instanceFactory;
};
#endif
