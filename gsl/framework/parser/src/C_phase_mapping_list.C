// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_phase_mapping_list.h"
#include "C_phase_mapping.h"
#include "DataItem.h"
#include <cassert>
#include "SyntaxError.h"

void C_phase_mapping_list::internalExecute(GslContext *c)
{
   std::vector<C_phase_mapping*>::iterator it, end = _phase_mappings->end();
   for (it = _phase_mappings->begin(); it != end; ++it) {
      (*it)->execute(c);
   }
}


C_phase_mapping_list::C_phase_mapping_list(const C_phase_mapping_list& rv)
   : C_production(rv), _phase_mappings(0)
{
   _phase_mappings = new std::vector<C_phase_mapping*>;
   if (rv._phase_mappings) {
      std::vector<C_phase_mapping*>::iterator it, 
	 end = rv._phase_mappings->end();
      for (it = rv._phase_mappings->begin(); it != end; ++it) {
	 _phase_mappings->push_back((*it)->duplicate());
      }
   }
}


C_phase_mapping_list::C_phase_mapping_list(C_phase_mapping *a, 
					   SyntaxError * error)
   : C_production(error), _phase_mappings(0)
{
   _phase_mappings = new std::vector<C_phase_mapping*>;
   _phase_mappings->push_back(a);
}


C_phase_mapping_list::C_phase_mapping_list(C_phase_mapping_list *al, 
					   C_phase_mapping *a, 
					   SyntaxError * error)
   : C_production(error), _phase_mappings(0)
{
   if (al) {
      if (al->isError()) {
	 delete _error;
	 _error = al->_error->duplicate();
      }
      std::unique_ptr<std::vector<C_phase_mapping*> > phases;
      al->releaseList(phases);
      _phase_mappings = phases.release();
      if (a) _phase_mappings->push_back(a);
   }
   delete al;
}


void C_phase_mapping_list::releaseList(
   std::unique_ptr<std::vector<C_phase_mapping*> >& phases)
{
   phases.reset(_phase_mappings);
   _phase_mappings = 0;
}

C_phase_mapping_list* C_phase_mapping_list::duplicate() const
{
   return new C_phase_mapping_list(*this);
}

C_phase_mapping_list::~C_phase_mapping_list()
{
   if (_phase_mappings) { 
      std::vector<C_phase_mapping*>::iterator it, end = _phase_mappings->end();
      for (it = _phase_mappings->begin(); it != end; ++it) {
         delete *it;
      }
   }
   delete _phase_mappings;
}

void C_phase_mapping_list::checkChildren() 
{
   if (_phase_mappings) {
      std::vector<C_phase_mapping*>::iterator it, end;
      end =_phase_mappings->end();
      for(it = _phase_mappings->begin(); it != end; ++it) {
	 (*it)->checkChildren();
	 if ((*it)->isError()) {
	    setError();
	 }
      }
   }
} 

void C_phase_mapping_list::recursivePrint() 
{
   if (_phase_mappings) {
      std::vector<C_phase_mapping*>::iterator it, end;
      end =_phase_mappings->end();
      for(it = _phase_mappings->begin(); it != end; ++it) {
	 (*it)->recursivePrint();
      }
   }
   printErrorMessage();
} 
