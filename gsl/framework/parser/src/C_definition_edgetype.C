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

#include "C_definition_edgetype.h"
#include "LensContext.h"
#include "C_argument_ndpair_clause_list.h"
#include "C_argument_declarator.h"
#include "C_ndpair_clause_list.h"
#include "C_declarator.h"
#include "C_argument.h"
#include "Simulation.h"
#include "EdgeType.h"
#include "EdgeTypeDataItem.h"
#include <memory>
#include "NDPair.h"
#include "NDPairListDataItem.h"
#include "DataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_phase_mapping_list.h"
#include "CompCategoryBase.h"   

#include <list>

void C_definition_edgetype::internalExecute(LensContext *c)
{
   _declarator->execute(c);

   if ( _argument )
      _argument->execute(c);

   // Checkout _arglist, if empty create an empty NDPairList and pass that 
   // to the getNodeType, if not if it is a declarator, search from the 
   // symbol table if it is a C_argument_ndpair_clause_list use that
   NDPairList localList;
   NDPairList *ndpList = &localList;

   if (_argument) {
      DataItem* di = _argument->getArgumentDataItem();
      const NDPairListDataItem* ndpdi = 
	 dynamic_cast<const NDPairListDataItem*>(di);
      if (ndpdi) {
	 NDPairList *tmp = ndpdi->getNDPairList();
	 if (tmp) {
	    ndpList = tmp;
	 }
      }
   }
   // get edgetype manager from simulation, create
   // edgetype dataitem and store in symbol table

   EdgeType* et = 0;
   try {
      et = c->sim->getEdgeType(_declarator->getName(), *ndpList);
   } catch (SyntaxErrorException& e) {
      throwError(e.getError());
   }
   if ( !et ) {
      std::string mes = "EdgeType " + _declarator->getName() + " not found";
      throwError(mes);
   }

   CompCategoryBase* ccBase = dynamic_cast<CompCategoryBase*>(et);
   if(ccBase) {
      c->setCurrentCompCategoryBase(ccBase);
      if (_phase_mapping_list) {
	 _phase_mapping_list->execute(c);
      }      
      ccBase->setUnmappedPhases(c);
#ifdef HAVE_MPI      
//      ccBase->setDistributionTemplates();             // commented out by Jizhu Lu on 02/15/2006
//      ccBase->setInitializationTemplates();             // added by Jizhu Lu on 04/06/2006
#endif // HAVE_MPI
   }

   EdgeTypeDataItem *etdi = new EdgeTypeDataItem();
   etdi->setEdgeType(et);

   std::auto_ptr<DataItem> etdi_ap(static_cast<DataItem*>(etdi));
   try {
      c->symTable.addEntry(_declarator->getName(), etdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While defining edgetype, " + e.getError());
   }
}


C_definition_edgetype* C_definition_edgetype::duplicate() const
{
   return new C_definition_edgetype(*this);
}


C_definition_edgetype::C_definition_edgetype(
   C_declarator *d, C_argument *a, C_phase_mapping_list* p,
   SyntaxError * error)
   : C_definition(error), _declarator(d), _argument(a), _phase_mapping_list(p)
{
}

C_definition_edgetype::C_definition_edgetype(const C_definition_edgetype& rv)
   : C_definition(rv), _declarator(0), _argument(0), _phase_mapping_list(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._argument) {
      _argument = rv._argument->duplicate();
   }
   if (rv._phase_mapping_list) {
      _phase_mapping_list = rv._phase_mapping_list->duplicate();
   }
}

C_definition_edgetype::~C_definition_edgetype()
{
   delete _declarator;
   delete _argument;
   delete _phase_mapping_list;
}

void C_definition_edgetype::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_argument) {
      _argument->checkChildren();
      if (_argument->isError()) {
         setError();
      }
   }
   if (_phase_mapping_list) {
      _phase_mapping_list->checkChildren();
      if (_phase_mapping_list->isError()) {
         setError();
      }
   }
} 

void C_definition_edgetype::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_argument) {
      _argument->recursivePrint();
   }
   if (_phase_mapping_list) {
      _phase_mapping_list->recursivePrint();
   }
   printErrorMessage();
} 
