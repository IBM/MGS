// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_logical_AND_expression.h"
#include "C_equality_expression.h"
#include "GridLayerDescriptor.h"
#include "Grid.h"
#include "SyntaxError.h"
#include "C_production_grid.h"
#include <algorithm>

void C_logical_AND_expression::internalExecute(GslContext *c, Grid* g)
{
   std::list<C_equality_expression*>::iterator 
      i = _listEqualityExpression->begin(), 
      end =_listEqualityExpression->end();

   C_equality_expression* ee = (*i++);
   ee->execute(c, g);
   _layers = ee->getLayers();
   _layers.sort();

   for (;i!=end;++i) {
      std::list<GridLayerDescriptor*>::iterator 
	 layers_begin = _layers.begin(), layers_end = _layers.end();

      ee = (*i);
      ee->execute(c, g);
      std::list<GridLayerDescriptor*> lin = ee->getLayers();
      lin.sort();
      std::list<GridLayerDescriptor*>::iterator lin_begin = lin.begin(), 
	 lin_end = lin.end();

      std::list<GridLayerDescriptor*> lout;
      std::insert_iterator<std::list<GridLayerDescriptor*> > lout_iter(
	 lout, lout.begin());

      set_intersection(layers_begin, layers_end, 
		       lin_begin, lin_end, lout_iter);
      lout.sort();
      _layers = lout;
   }
   _layers.unique();
}


const std::list<GridLayerDescriptor*>& 
C_logical_AND_expression::getLayers() const
{
   return _layers;
}


C_logical_AND_expression::C_logical_AND_expression(
   const C_logical_AND_expression& rv)
   : C_production_grid(rv), _listEqualityExpression(0), _layers(rv._layers)
{
   if(rv._listEqualityExpression) {
      _listEqualityExpression = new std::list<C_equality_expression*>;
      std::list<C_equality_expression*>::iterator i, 
	 end = rv._listEqualityExpression->end();
      for (i = rv._listEqualityExpression->begin(); i != end; ++i) {
         _listEqualityExpression->push_back((*i)->duplicate());
      }
   }

}


C_logical_AND_expression::C_logical_AND_expression(
   C_logical_AND_expression *lae, C_equality_expression *ee, 
   SyntaxError * error)
   : C_production_grid(error), _listEqualityExpression(0)
{
   _listEqualityExpression = lae->releaseSet();
   _listEqualityExpression->push_back(ee);
   delete lae;
}


C_logical_AND_expression::C_logical_AND_expression(
   C_equality_expression *ee, SyntaxError * error)
   : C_production_grid(error), _listEqualityExpression(0)
{
   _listEqualityExpression = new std::list<C_equality_expression*>;
   _listEqualityExpression->push_back(ee);
}


std::list<C_equality_expression*>* C_logical_AND_expression::releaseSet()
{
   std::list<C_equality_expression*> *retval = _listEqualityExpression;
   _listEqualityExpression = 0;
   return retval;
}


C_logical_AND_expression* C_logical_AND_expression::duplicate() const
{
   return new C_logical_AND_expression(*this);
}


C_logical_AND_expression::~C_logical_AND_expression()
{
   if (_listEqualityExpression) {
      std::list<C_equality_expression*>::iterator i, 
	 end =_listEqualityExpression->end();
      for (i = _listEqualityExpression->begin(); i != end; ++i) {
	 delete (*i);
      }
   }
   delete _listEqualityExpression;
}

void C_logical_AND_expression::checkChildren() 
{
   if (_listEqualityExpression) {
      std::list<C_equality_expression*>::iterator i, begin, end;
      begin =_listEqualityExpression->begin();
      end =_listEqualityExpression->end();
      for(i=begin;i!=end;++i) {
	 (*i)->checkChildren();
	 if ((*i)->isError()) {
	    setError();
	 }
      }
   }
} 

void C_logical_AND_expression::recursivePrint() 
{
   if (_listEqualityExpression) {
      std::list<C_equality_expression*>::iterator i, begin, end;
      begin =_listEqualityExpression->begin();
      end =_listEqualityExpression->end();
      for(i=begin;i!=end;++i) {
	 (*i)->recursivePrint();
      }
   }
   printErrorMessage();
} 
