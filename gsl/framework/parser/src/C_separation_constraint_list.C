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

#include "C_separation_constraint_list.h"
#include "C_separation_constraint.h"
#include "DataItem.h"
#include <cassert>
#include "SyntaxError.h"

void C_separation_constraint_list::internalExecute(LensContext *c)
{
   std::vector<C_separation_constraint*>::iterator it, 
      end = _separation_constraints->end();
   for (it = _separation_constraints->begin(); it != end; ++it) {
      (*it)->execute(c);
   }
}


C_separation_constraint_list::C_separation_constraint_list(
   const C_separation_constraint_list& rv)
   : C_production(rv), _separation_constraints(0)
{
   _separation_constraints = new std::vector<C_separation_constraint*>;
   if (rv._separation_constraints) {
      std::vector<C_separation_constraint*>::iterator it, 
	 end = rv._separation_constraints->end();
      for (it = rv._separation_constraints->begin(); it != end; ++it) {
	 _separation_constraints->push_back((*it)->duplicate());
      }
   }
}


C_separation_constraint_list::C_separation_constraint_list(
   C_separation_constraint *a, SyntaxError * error)
   : C_production(error), _separation_constraints(0)
{
   _separation_constraints = new std::vector<C_separation_constraint*>;
   _separation_constraints->push_back(a);
}


C_separation_constraint_list::C_separation_constraint_list(
   C_separation_constraint_list *al, C_separation_constraint *a, 
   SyntaxError * error)
   : C_production(error), _separation_constraints(0)
{
   if (al) {
      if (al->isError()) {
	 delete _error;
	 _error = al->_error->duplicate();
      }
      std::auto_ptr<std::vector<C_separation_constraint*> > 
	 separation_constraints;
      al->releaseList(separation_constraints);
      _separation_constraints = separation_constraints.release();
      if (a) _separation_constraints->push_back(a);
   }
   delete al;
}


void C_separation_constraint_list::releaseList(
   std::auto_ptr<std::vector<C_separation_constraint*> >& 
   separation_constraints)
{
   separation_constraints.reset(_separation_constraints);
   _separation_constraints = 0;
}

C_separation_constraint_list* C_separation_constraint_list::duplicate() const
{
   return new C_separation_constraint_list(*this);
}

C_separation_constraint_list::~C_separation_constraint_list()
{
   if (_separation_constraints) { 
      std::vector<C_separation_constraint*>::iterator it, 
	 end = _separation_constraints->end();
      for (it = _separation_constraints->begin(); it != end; ++it) {
         delete *it;
      }
   }
   delete _separation_constraints;
}

void C_separation_constraint_list::checkChildren() 
{
   if (_separation_constraints) {
      std::vector<C_separation_constraint*>::iterator it, end;
      end =_separation_constraints->end();
      for(it = _separation_constraints->begin(); it != end; ++it) {
	 (*it)->checkChildren();
	 if ((*it)->isError()) {
	    setError();
	 }
      }
   }
} 

void C_separation_constraint_list::recursivePrint() 
{
   if (_separation_constraints) {
      std::vector<C_separation_constraint*>::iterator it, end;
      end =_separation_constraints->end();
      for(it = _separation_constraints->begin(); it != end; ++it) {
	 (*it)->recursivePrint();
      }
   }
   printErrorMessage();
} 
