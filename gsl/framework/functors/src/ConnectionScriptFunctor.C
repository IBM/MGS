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

#include "ConnectionScriptFunctor.h"
#include "LensContext.h"
#include"C_connection_script_definition_body.h"
#include "DataItem.h"
#include "InstanceFactoryQueriable.h"
#include "SyntaxErrorException.h"

ConnectionScriptFunctor::ConnectionScriptFunctor(
   C_connection_script_definition_body *db, std::list<C_parameter_type> *ptl)
{
   _def_body = db->duplicate();
   _paramTypeList = new std::list<C_parameter_type>(*ptl);
}


ConnectionScriptFunctor::ConnectionScriptFunctor(
   const ConnectionScriptFunctor& csf)
{
   _def_body = csf._def_body->duplicate();
   _paramTypeList = new std::list<C_parameter_type>(*csf._paramTypeList);
}


void ConnectionScriptFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new ConnectionScriptFunctor(*this));
}


ConnectionScriptFunctor::~ConnectionScriptFunctor()
{
   delete _def_body;
   delete _paramTypeList;
}


void ConnectionScriptFunctor::doInitialize(LensContext *c, 
					   const std::vector<DataItem*>& args)
{
}


void ConnectionScriptFunctor::doExecute(LensContext *c, 
					const std::vector<DataItem*>& args, 
					std::auto_ptr<DataItem>& rvalue)
{
   c->symTable.addLocalScope();
   DataItem *currentArg = 0;
   std::string currentName("");
   int argListSize = args.size();
   int paramListSize = _paramTypeList->size();
   if (argListSize < paramListSize) {
      throw SyntaxErrorException(
	 "Too few arguments calling connection script");
   }
   std::vector<DataItem*>::const_iterator a, abegin = args.begin();

   std::list<C_parameter_type>::iterator p,pbegin = _paramTypeList->begin();
   std::list<C_parameter_type>::iterator pend = _paramTypeList->end();
   for (a=abegin,p=pbegin;p!=pend;++p, ++a) {
      if (!(*p).isSpecified()) break;
      currentArg = *a;
      currentName = p->getIdentifier();
      // Do type checking here

      // Now put named arguments into the symbol table
      if (currentName!="") {
         std::auto_ptr<DataItem> diap;
         currentArg->duplicate(diap);
         c->symTable.addEntry(currentName,diap);
      }

   }
   // Copy the definition body and execute the copy so that 
   // The state is gotten rid of the next time the functor is called.
   C_connection_script_definition_body *localCopy = 
      new C_connection_script_definition_body(*_def_body);
   try {
      localCopy->execute(c);
   } catch (SyntaxErrorException& e) {
      e.printError();
      e.resetError();
      localCopy->recursivePrint();
      localCopy->printTdError();
      throw;
   }
   delete localCopy;
   c->symTable.removeLocalScope();
}

