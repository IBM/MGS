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
      virtual void duplicate(std::auto_ptr<Functor> &fap) const;
      virtual ~ConnectionScriptFunctor();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args);
      virtual void doExecute(LensContext *c, 
			     const std::vector<DataItem*>& args, 
			     std::auto_ptr<DataItem>& rvalue);
   private:

      C_connection_script_definition_body *_def_body;
      std::list<C_parameter_type> *_paramTypeList;
};
#endif
