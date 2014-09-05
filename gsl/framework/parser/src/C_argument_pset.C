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

#include "C_argument_pset.h"
#include "LensContext.h"
#include "C_parameter_type_pair.h"
#include "C_ndpair_clause_list.h"
#include "ParameterSet.h"
#include "ParameterSetDataItem.h"
#include <memory>
#include "NodeType.h"
#include "EdgeType.h"
#include "Simulation.h"
#include "NDPairList.h"
#include <list>
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_argument_pset::internalExecute(LensContext *c)
{
   _parm_type_pair->execute(c);
   _ndp_clause_list->execute(c);
   _parameter_set_DI = new ParameterSetDataItem();
   NDPairList dummy;

   std::auto_ptr<ParameterSet> pset;
   if (_parm_type_pair->getModelType() == C_parameter_type_pair::_EDGE) {
      EdgeType* et = 0;
      if (_ndp_clause_list) {
	 NDPairList* ndpList = _ndp_clause_list->getList();
	 if (ndpList) {
	    et = c->sim->getEdgeType(
	       _parm_type_pair->getModelName(), *ndpList);
	 } 
      } 
      if (et == 0) {
	 et = c->sim->getEdgeType(_parm_type_pair->getModelName(), dummy);
      }
      if (_parm_type_pair->getParameterType() == C_parameter_type_pair::_INIT)
         et->getInitializationParameterSet(pset);
   }
   else if (_parm_type_pair->getModelType() == C_parameter_type_pair::_NODE) {
      NodeType* nt = c->sim->getNodeType(
	 _parm_type_pair->getModelName(), dummy);
      if (_parm_type_pair->getParameterType() == C_parameter_type_pair::_INIT)
         nt->getInitializationParameterSet(pset);
      else if (_parm_type_pair->getParameterType() == 
	       C_parameter_type_pair::_IN)
         nt->getInAttrParameterSet(pset);
      else if (_parm_type_pair->getParameterType() == 
	       C_parameter_type_pair::_OUT)
         nt->getOutAttrParameterSet(pset);
   }

   try {
      pset->set(*(_ndp_clause_list->getList()));
   } catch (SyntaxErrorException& e) {
      throwError(e.getError());
   }
   _parameter_set_DI->setParameterSet(pset);
}


C_argument_pset::C_argument_pset(const C_argument_pset& rv)
   : C_argument(rv), _parm_type_pair(0), _ndp_clause_list(0), 
     _parameter_set_DI(0)
{
   if (rv._parm_type_pair) {
      _parm_type_pair = rv._parm_type_pair->duplicate();
   }
   if (rv._ndp_clause_list) 
      _ndp_clause_list = rv._ndp_clause_list->duplicate();
   if (rv._parameter_set_DI) {
      std::auto_ptr<DataItem> cc_di;
      rv._parameter_set_DI->duplicate(cc_di);
      _parameter_set_DI = dynamic_cast<ParameterSetDataItem*>(cc_di.release());
   }
}


C_argument_pset::C_argument_pset(
   C_parameter_type_pair *p, C_ndpair_clause_list *n, SyntaxError * error)
   : C_argument(_PSET, error), _parm_type_pair(p), _ndp_clause_list(n), 
     _parameter_set_DI(0)
{
}


C_argument_pset* C_argument_pset::duplicate() const
{
   return new C_argument_pset(*this);
}


C_argument_pset::~C_argument_pset()
{
   delete _parm_type_pair;
   delete _ndp_clause_list;
   delete _parameter_set_DI;
}


DataItem* C_argument_pset::getArgumentDataItem() const
{
   return _parameter_set_DI;
}

void C_argument_pset::checkChildren() 
{
   if (_parm_type_pair) {
      _parm_type_pair->checkChildren();
      if (_parm_type_pair->isError()) {
         setError();
      }
   }
   if (_ndp_clause_list) {
      _ndp_clause_list->checkChildren();
      if (_ndp_clause_list->isError()) {
         setError();
      }
   }
} 

void C_argument_pset::recursivePrint() 
{
   if (_parm_type_pair) {
      _parm_type_pair->recursivePrint();
   }
   if (_ndp_clause_list) {
      _ndp_clause_list->recursivePrint();
   }
   printErrorMessage();
} 
