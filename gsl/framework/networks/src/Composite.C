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

#include "Composite.h"
#include "Repertoire.h"

Composite::Composite()
: _repertoire(0)
{

}


Repertoire* Composite::getRepertoire() const
{
   return _repertoire;
}


void Composite::setRepertoire(Repertoire* repertoire)
{
   _repertoire = repertoire;

}


Composite::~Composite()
{
   delete _repertoire;
}
