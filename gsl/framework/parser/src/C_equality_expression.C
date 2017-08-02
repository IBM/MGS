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

#include "C_equality_expression.h"
#include "C_primary_expression.h"
#include "C_name.h"
#include "GridLayerDescriptor.h"
#include "Grid.h"
#include "SyntaxError.h"
#include "C_production_grid.h"
#include "NDPair.h"
#include "NDPairList.h"

void C_equality_expression::internalExecute(LensContext *c, Grid* g)
{
   _layers.clear();
   if(_primaryExpression) {
      _primaryExpression->execute(c, g);
      _layers = _primaryExpression->getLayers();
   }
   if(_name) {
      _name->execute(c);

      std::vector<GridLayerDescriptor*> const & glayers = g->getLayers();
      std::string name = _name->getName();

      std::vector<GridLayerDescriptor*>::const_iterator gld_iter, 
	 gld_end = glayers.end();

      for (gld_iter = glayers.begin(); gld_iter != gld_end; ++gld_iter) {
         GridLayerDescriptor* gld = (*gld_iter);
         const NDPairList& ndpList = gld->getNDPList();
         NDPairList::const_iterator ndp_iter, ndp_end = ndpList.end();
         for (ndp_iter = ndpList.begin(); ndp_iter != ndp_end; ++ndp_iter) {
            NDPair* ndp = (*ndp_iter);
            if (name == ndp->getName()) {
               std::string v = ndp->getValue();
               if ( (_equivalence && (v == *_value)) || 
		    (!_equivalence && (v != *_value)) )
                  _layers.push_back(gld);
            }
         }
      }
      _layers.sort();
   }
}


const std::list<GridLayerDescriptor*>& C_equality_expression::getLayers() const
{
   return _layers;
}


C_equality_expression::C_equality_expression(const C_equality_expression& rv)
   : C_production_grid(rv), _equivalence(rv._equivalence), _layers(rv._layers),
     _name(0), _primaryExpression(0), _value(0)
{
   if(rv._primaryExpression) {
      _primaryExpression = rv._primaryExpression->duplicate();
   }
   if(rv._name) {
      _name = rv._name->duplicate();
   }
   if (rv._value) {
      _value = new std::string(*(rv._value));
   }
}


C_equality_expression::C_equality_expression(
   C_primary_expression *pe, SyntaxError * error)
   : C_production_grid(error), _equivalence(false), _name(0), 
     _primaryExpression(pe), _value(0)
{
   _value = new std::string("");
}


C_equality_expression::C_equality_expression(
   C_name *n, std::string *s, bool equiv, SyntaxError * error)
   : C_production_grid(error), _equivalence(equiv), _name(n), 
     _primaryExpression(0), _value(s)
{
}


C_equality_expression* C_equality_expression::duplicate() const
{
   return new C_equality_expression(*this);
}


C_equality_expression::~C_equality_expression()
{
   delete _primaryExpression;
   delete _name;
   delete _value;
}

void C_equality_expression::checkChildren() 
{
   if (_name) {
      _name->checkChildren();
      if (_name->isError()) {
         setError();
      }
   }
   if (_primaryExpression) {
      _primaryExpression->checkChildren();
      if (_primaryExpression->isError()) {
         setError();
      }
   }
} 

void C_equality_expression::recursivePrint() 
{
   if (_name) {
      _name->recursivePrint();
   }
   if (_primaryExpression) {
      _primaryExpression->recursivePrint();
   }
   printErrorMessage();
} 
