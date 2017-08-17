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

#include "C_preamble.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_preamble::internalExecute(LensContext *c)
{
}

C_preamble::C_preamble(const C_preamble& rv)
   : C_production(rv), _listStrings(0)
{
   if (rv._listStrings) {
      _listStrings = new std::list<std::string> (*rv._listStrings);
   }
}


C_preamble::C_preamble(SyntaxError * error)
   : C_production(error), _listStrings(0)
{
   _listStrings = new std::list<std::string>;
}


C_preamble::C_preamble(std::string *n, SyntaxError * error)
   : C_production(error), _listStrings(0)
{
   _listStrings = new std::list<std::string>;
   _listStrings->push_back(*n);
   delete n;
}


C_preamble::C_preamble(C_preamble *pr, std::string *n, SyntaxError * error)
   : C_production(error), _listStrings(0)
{
   if (pr) {
      if (pr->isError()) {
	 delete _error;
	 _error = pr->_error->duplicate();
      }
      _listStrings = pr->releaseList();
      if (n) _listStrings->push_back(*n);
   }
   delete pr;
   delete n;
}


std::list<std::string>* C_preamble::releaseList()
{
   std::list<std::string> *retval = _listStrings;
   _listStrings = 0;
   return retval;
}


C_preamble* C_preamble::duplicate() const
{
   return new C_preamble(*this);
}


std::list<std::string>* C_preamble::getListStrings() const
{
   return _listStrings;
}


C_preamble::~C_preamble()
{
   //   delete _preamble;
   delete _listStrings;
}

void C_preamble::checkChildren() 
{
} 

void C_preamble::recursivePrint() 
{
   printErrorMessage();
} 
