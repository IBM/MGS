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

#include "C_phase_list.h"
#include "C_phase.h"
#include "DataItem.h"
#include <cassert>
#include "SyntaxError.h"

void C_phase_list::internalExecute(LensContext *c)
{
   std::vector<C_phase*>::iterator it, end = _phases->end();
   for (it = _phases->begin(); it != end; ++it) {
      (*it)->execute(c);
   }
}


C_phase_list::C_phase_list(const C_phase_list& rv)
   : C_production(rv), _phases(0)
{
   _phases = new std::vector<C_phase*>;
   if (rv._phases) {
      std::vector<C_phase*>::iterator it, end = rv._phases->end();
      for (it = rv._phases->begin(); it != end; ++it) {
	 _phases->push_back((*it)->duplicate());
      }
   }
}


C_phase_list::C_phase_list(C_phase *a, SyntaxError * error)
   : C_production(error), _phases(0)
{
   _phases = new std::vector<C_phase*>;
   _phases->push_back(a);
}


C_phase_list::C_phase_list(C_phase_list *al, C_phase *a, SyntaxError * error)
   : C_production(error), _phases(0)
{
   if (al) {
      if (al->isError()) {
	 delete _error;
	 _error = al->_error->duplicate();
      }
      std::unique_ptr<std::vector<C_phase*> > phases;
      al->releaseList(phases);
      _phases = phases.release();
      if (a) _phases->push_back(a);
   }
   delete al;
}


void C_phase_list::releaseList(
   std::unique_ptr<std::vector<C_phase*> >& phases)
{
   phases.reset(_phases);
   _phases = 0;
}

C_phase_list* C_phase_list::duplicate() const
{
   return new C_phase_list(*this);
}

C_phase_list::~C_phase_list()
{
   if (_phases) { 
      std::vector<C_phase*>::iterator it, end = _phases->end();
      for (it = _phases->begin(); it != end; ++it) {
         delete *it;
      }
   }
   delete _phases;
}

void C_phase_list::checkChildren() 
{
   if (_phases) {
      std::vector<C_phase*>::iterator it, end;
      end =_phases->end();
      for(it = _phases->begin(); it != end; ++it) {
	 (*it)->checkChildren();
	 if ((*it)->isError()) {
	    setError();
	 }
      }
   }
} 

void C_phase_list::recursivePrint() 
{
   if (_phases) {
      std::vector<C_phase*>::iterator it, end;
      end =_phases->end();
      for(it = _phases->begin(); it != end; ++it) {
	 (*it)->recursivePrint();
      }
   }
   printErrorMessage();
} 
