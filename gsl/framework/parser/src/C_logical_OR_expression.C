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

#include "C_logical_OR_expression.h"
#include "C_logical_AND_expression.h"
#include "GridLayerDescriptor.h"
#include "Grid.h"
#include "SyntaxError.h"
#include "C_production_grid.h"

void C_logical_OR_expression::internalExecute(LensContext *c, Grid* g)
{
   _layers.sort();

   std::list<C_logical_AND_expression*>::iterator i, 
      end =_listLogicalAndExpression->end();
   for (i = _listLogicalAndExpression->begin(); i != end; ++i) {
      C_logical_AND_expression* lae = (*i);
      lae->execute(c, g);
      std::list<GridLayerDescriptor*> lin = lae->getLayers();
      lin.sort();
      _layers.merge(lin);
   }
   _layers.unique();
}


const std::list<GridLayerDescriptor*>& 
C_logical_OR_expression::getLayers() const
{
   return _layers;
}


C_logical_OR_expression::C_logical_OR_expression(
   const C_logical_OR_expression& rv)
   : C_production_grid(rv), _listLogicalAndExpression(0), _layers(rv._layers)
{
   _listLogicalAndExpression = new std::list<C_logical_AND_expression*>;
   if(rv._listLogicalAndExpression) {
      std::list<C_logical_AND_expression*>::iterator i, 
	 end = rv._listLogicalAndExpression->end();
      for (i = rv._listLogicalAndExpression->begin(); i != end; ++i) {
         _listLogicalAndExpression->push_back((*i)->duplicate());
      }
   }

}


C_logical_OR_expression::C_logical_OR_expression(
   C_logical_OR_expression *loe, C_logical_AND_expression *lae, 
   SyntaxError * error)
   : C_production_grid(error), _listLogicalAndExpression(0)
{
   _listLogicalAndExpression = loe->releaseSet();
   _listLogicalAndExpression->push_back(lae);
   delete loe;
}


C_logical_OR_expression::C_logical_OR_expression(
   C_logical_AND_expression *lae, SyntaxError * error)
   : C_production_grid(error), _listLogicalAndExpression(0)
{
   _listLogicalAndExpression = new std::list<C_logical_AND_expression*>;
   _listLogicalAndExpression->push_back(lae);
}


std::list<C_logical_AND_expression*>* C_logical_OR_expression::releaseSet()
{
   std::list<C_logical_AND_expression*>* retval = _listLogicalAndExpression;
   _listLogicalAndExpression = 0;
   return retval;
}


C_logical_OR_expression* C_logical_OR_expression::duplicate() const
{
   return new C_logical_OR_expression(*this);
}


C_logical_OR_expression::~C_logical_OR_expression()
{
   if (_listLogicalAndExpression) {
      std::list<C_logical_AND_expression*>::iterator i, 
	 end =_listLogicalAndExpression->end();
      for (i=_listLogicalAndExpression->begin(); i != end; ++i) {
	 delete (*i);
      }
   }
   delete _listLogicalAndExpression;
}

void C_logical_OR_expression::checkChildren() 
{
   if (_listLogicalAndExpression) {
      std::list<C_logical_AND_expression*>::iterator i, begin, end;
      begin =_listLogicalAndExpression->begin();
      end =_listLogicalAndExpression->end();
      for(i=begin;i!=end;++i) {
	 (*i)->checkChildren();
	 if ((*i)->isError()) {
	    setError();
	 }
      }
   }
} 

void C_logical_OR_expression::recursivePrint() 
{
   if (_listLogicalAndExpression) {
      std::list<C_logical_AND_expression*>::iterator i, begin, end;
      begin =_listLogicalAndExpression->begin();
      end =_listLogicalAndExpression->end();
      for(i=begin;i!=end;++i) {
	 (*i)->recursivePrint();
      }
   }
   printErrorMessage();
} 
