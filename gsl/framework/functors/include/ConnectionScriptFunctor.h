// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef CONNECTIONSCRIPTFUNCTOR_H
#include "Copyright.h"
#define CONNECTIONSCRIPTFUNCTOR_H

#include "ConnectorFunctor.h"
#include "C_parameter_type.h"
#include <memory>
#include <list>
#include <vector>
class C_connection_script_definition_body;
class DataItem;
class LensContext;

class ConnectionScriptFunctor: public ConnectorFunctor
{
   public:
      ConnectionScriptFunctor(C_connection_script_definition_body *, 
			      std::list<C_parameter_type> *);
      ConnectionScriptFunctor(const ConnectionScriptFunctor&);
      virtual void duplicate(std::unique_ptr<Functor>&& fap) const;
      virtual ~ConnectionScriptFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::unique_ptr<DataItem>& rvalue);
   private:

      C_connection_script_definition_body *_def_body;
      std::list<C_parameter_type> *_paramTypeList;
};
#endif
