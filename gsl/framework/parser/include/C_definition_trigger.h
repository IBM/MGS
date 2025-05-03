// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_DEFINITION_TRIGGER_H
#define C_DEFINITION_TRIGGER_H
#include "Copyright.h"

#include <string>
#include <list>

#include "C_definition.h"

class C_declarator;
class C_parameter_type;
class C_parameter_type_list;
class C_trigger;
class GslContext;
class InstanceFactory;
class SyntaxError;

class C_definition_trigger : public C_definition
{
   public:
      C_definition_trigger(const C_definition_trigger&);
      C_definition_trigger(C_declarator *, C_parameter_type_list *, 
			   SyntaxError *);
      virtual ~C_definition_trigger();
      virtual C_definition_trigger* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessor methods
      std::string getDeclarator();
      std::list<C_parameter_type>* getConstructorParams();
      InstanceFactory* getInstanceFactory();

   private:
      C_declarator* _declarator;
      C_parameter_type_list* _constructor_ptl;
      InstanceFactory* _instanceFactory;
      std::list<C_parameter_type>* _constructor_list;
};
#endif
