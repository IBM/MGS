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

#include "C_ndpair_clause.h"
#include "NDPair.h"
#include "C_name.h"
#include "C_argument.h"

void C_ndpair_clause::internalExecute(LensContext *c)
{
   _name->execute(c);
   _argument->execute(c);
   delete _ndpair;
   std::unique_ptr<DataItem> di;
   _argument->getArgumentDataItem()->duplicate(di);
   _ndpair = new NDPair(_name->getName(), di);
}


C_ndpair_clause::C_ndpair_clause(const C_ndpair_clause& rv)
   : C_production(rv), _ndpair(0), _name(0), _argument(0)
{
   if (rv._name) {
      _name = rv._name->duplicate();
   }
   if (rv._argument) {
      _argument = rv._argument->duplicate();
   }
}


C_ndpair_clause::C_ndpair_clause(C_name *nm, C_argument *ar, 
				 SyntaxError * error)
   : C_production(error), _ndpair(0), _name(nm), _argument(ar)
{
}


C_ndpair_clause* C_ndpair_clause::duplicate() const
{
   return new C_ndpair_clause(*this);
}

void C_ndpair_clause::releaseNDPair(std::unique_ptr<NDPair>& ndp)
{
   ndp.reset(_ndpair);
   _ndpair = 0;
}

C_ndpair_clause::~C_ndpair_clause()
{
   delete _name;
   delete _argument;
   delete _ndpair;
}

void C_ndpair_clause::checkChildren() 
{
   if (_name) {
      _name->checkChildren();
      if (_name->isError()) {
         setError();
      }
   }
   if (_argument) {
      _argument->checkChildren();
      if (_argument->isError()) {
         setError();
      }
   }
} 

void C_ndpair_clause::recursivePrint() 
{
   if (_name) {
      _name->recursivePrint();
   }
   if (_argument) {
      _argument->recursivePrint();
   }
   printErrorMessage();
} 
