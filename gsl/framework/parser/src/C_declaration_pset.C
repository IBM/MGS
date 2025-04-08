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

#include "C_declaration_pset.h"
#include "LensContext.h"
#include "C_parameter_type_pair.h"
#include "C_declarator.h"
#include "C_ndpair_clause_list.h"
#include "NodeType.h"
#include "EdgeType.h"
#include "ParameterSet.h"
#include "ParameterSetDataItem.h"
#include "Simulation.h"
#include "NDPairList.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

#include <memory>

void C_declaration_pset::internalExecute(LensContext *c)
{
   _parameterTypePair->execute(c);
   _declarator->execute(c);
   _ndpClauseList->execute(c);
   NDPairList dummy;

   std::unique_ptr<ParameterSet> pset;
   if (_parameterTypePair->getModelType() == C_parameter_type_pair::_EDGE) {
      EdgeType* et = 0;
      if (_ndpClauseList) {
	 NDPairList* ndpList = _ndpClauseList->getList();
	 if (ndpList) {
	    et = c->sim->getEdgeType(
	       _parameterTypePair->getModelName(), *ndpList);
	 }
      } 
      if (et == 0) {
	 et = c->sim->getEdgeType(
	    _parameterTypePair->getModelName(), dummy);
      }
      if (_parameterTypePair->getParameterType() == 
	  C_parameter_type_pair::_INIT)
         et->getInitializationParameterSet(pset);
   }
   else if (_parameterTypePair->getModelType() == 
	    C_parameter_type_pair::_NODE) {
      NodeType* nt = c->sim->getNodeType(
	 _parameterTypePair->getModelName(), dummy);
      if (_parameterTypePair->getParameterType() == 
	  C_parameter_type_pair::_INIT)
         nt->getInitializationParameterSet(std::move(pset));
      else if (_parameterTypePair->getParameterType() == 
	       C_parameter_type_pair::_IN)
         nt->getInAttrParameterSet(std::move(pset));
      else if (_parameterTypePair->getParameterType() == 
	       C_parameter_type_pair::_OUT)
         nt->getOutAttrParameterSet(std::move(pset));
   }

   try {
      pset->set(*(_ndpClauseList->getList()));
   } catch (SyntaxErrorException& e) {
      throwError(e.getError());
   }

   ParameterSetDataItem* psdi = new ParameterSetDataItem();
   psdi->setParameterSet(pset);
   std::unique_ptr<DataItem> psdi_ap(psdi);
   try {
      c->symTable.addEntry(_declarator->getName(), psdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring pset, " + e.getError());
   }
}


C_declaration_pset* C_declaration_pset::duplicate() const
{
   return new C_declaration_pset(*this);
}


C_declaration_pset::C_declaration_pset(
   C_parameter_type_pair *p, C_declarator *d, 
   C_ndpair_clause_list *n, SyntaxError * error)
   : C_declaration(error), _parameterTypePair(p), _declarator(d), 
     _ndpClauseList(n)
{
}


C_declaration_pset::C_declaration_pset(const C_declaration_pset& rv)
   : C_declaration(rv), _parameterTypePair(0), _declarator(0), 
     _ndpClauseList(0)
{
   if (rv._parameterTypePair) {
      _parameterTypePair = rv._parameterTypePair->duplicate();
   }
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._ndpClauseList) {
      _ndpClauseList = rv._ndpClauseList->duplicate();
   }
}


C_declaration_pset::~C_declaration_pset()
{
   delete _parameterTypePair;
   delete _declarator;
   delete _ndpClauseList;
}

void C_declaration_pset::checkChildren() 
{
   if (_parameterTypePair) {
      _parameterTypePair->checkChildren();
      if (_parameterTypePair->isError()) {
         setError();
      }
   }
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_ndpClauseList) {
      _ndpClauseList->checkChildren();
      if (_ndpClauseList->isError()) {
         setError();
      }
   }
} 

void C_declaration_pset::recursivePrint() 
{
   if (_parameterTypePair) {
      _parameterTypePair->recursivePrint();
   }
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_ndpClauseList) {
      _ndpClauseList->recursivePrint();
   }
   printErrorMessage();
} 
