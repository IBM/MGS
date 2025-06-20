// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _SCRIPTFUNCTORTYPE_H_
#define _SCRIPTFUNCTORTYPE_H_
#include "Copyright.h"

#include "FunctorType.h"
#include <memory>
#include <string>

class C_connection_script_definition;

class ScriptFunctorType : public FunctorType
{
   public:
      ScriptFunctorType (ScriptFunctorType const *);
      ScriptFunctorType (C_connection_script_definition *, std::string const &name);
      virtual ~ScriptFunctorType ();

      // FunctorType methods
      void getFunctor(std::unique_ptr<Functor> & r_aptr);
      Functor* getFunctor();
      virtual std::string getName();
      virtual std::string getDescription();
      virtual void getQueriable(std::unique_ptr<InstanceFactoryQueriable>& dup);
   private:
      std::string _functorName;
      C_connection_script_definition *_c_script_def;
};
#endif
