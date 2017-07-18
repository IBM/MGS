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

#ifndef C_DEFINITION_CONSTANTTYPE_H
#define C_DEFINITION_CONSTANTTYPE_H
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

class C_definition_constanttype : public C_definition
{
   public:
      C_definition_constanttype(const C_definition_constanttype&);
      C_definition_constanttype(C_declarator *, SyntaxError *);
      virtual ~C_definition_constanttype();
      virtual C_definition_constanttype* duplicate() const;
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
