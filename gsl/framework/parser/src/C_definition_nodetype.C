// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_definition_nodetype.h"
#include "C_declarator.h"
#include "C_argument.h"
#include "C_argument_ndpair_clause_list.h"
#include "C_argument_declarator.h"
#include "C_ndpair_clause_list.h"
#include "LensContext.h"
#include "NodeType.h"
#include "NodeTypeDataItem.h"
#include "Simulation.h"
#include "NDPair.h"
#include "NDPairListDataItem.h"
#include "DataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_phase_mapping_list.h"
//#include "CompCategoryBase.h"
#include "DistributableCompCategoryBase.h"    // modified by Jizhu Lu on 02/15/2006
 
#include <memory>
#include <list>

void C_definition_nodetype::internalExecute(LensContext *c)
{
   _declarator->execute(c);
   
   if ( _argument ) {
      _argument->execute(c);
   }

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
	 NDPairList* tmp = ndpdi->getNDPairList();
	 if (tmp) {
	    ndpList = tmp;
	 }
      }
   }

   // Get nodetype manager from simulation, create
   // nodetype dataitem and store in symbol table

   NodeType* nt = 0;
   try {
      nt = c->sim->getNodeType(_declarator->getName(), *ndpList);
   } catch (SyntaxErrorException& e) {
      throwError(e.getError());
   }
   if( !nt ) {
      std::string mes = "Node type " + _declarator->getName() 
	 + " does not correspond to a valid type";
      throwError(mes);
   }

//   CompCategoryBase* ccBase = dynamic_cast<CompCategoryBase*>(nt);
   DistributableCompCategoryBase* ccBase = dynamic_cast<DistributableCompCategoryBase*>(nt);  // modified by Jizhu Lu on 02/15/2006
   if(ccBase) {
      c->setCurrentCompCategoryBase(ccBase);
      if (_phase_mapping_list) {
	 _phase_mapping_list->execute(c);
      }      
      ccBase->setUnmappedPhases(c);
#ifdef HAVE_MPI      
      ccBase->setDistributionTemplates();
#endif // HAVE_MPI
   }

   NodeTypeDataItem *ntdi = new NodeTypeDataItem();
   ntdi->setNodeType(nt);

   std::auto_ptr<DataItem> ntdi_ap(static_cast<DataItem*>(ntdi));
   try {
      c->symTable.addEntry(_declarator->getName(), ntdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While defining nodetype, " + e.getError());
   }
}

C_definition_nodetype* C_definition_nodetype::duplicate() const
{
   return new C_definition_nodetype(*this);
}

C_definition_nodetype::C_definition_nodetype(
   C_declarator *d, C_argument *a, C_phase_mapping_list* p,
   SyntaxError * error)
   : C_definition(error), _declarator(d), _argument(a), _phase_mapping_list(p)
{
}

C_definition_nodetype::C_definition_nodetype(const C_definition_nodetype& rv)
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

C_definition_nodetype::~C_definition_nodetype()
{
   delete _declarator;
   delete _argument;
   delete _phase_mapping_list;
}

void C_definition_nodetype::checkChildren() 
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

void C_definition_nodetype::recursivePrint() 
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
