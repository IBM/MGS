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
      void getFunctor(std::auto_ptr<Functor> & r_aptr);
      Functor* getFunctor();
      virtual std::string getName();
      virtual std::string getDescription();
      virtual void getQueriable(std::auto_ptr<InstanceFactoryQueriable>& dup);
   private:
      std::string _functorName;
      C_connection_script_definition *_c_script_def;
};
#endif
