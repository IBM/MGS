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
class LensContext;
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
      virtual void internalExecute(LensContext *);
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
