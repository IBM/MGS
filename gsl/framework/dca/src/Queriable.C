// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "Queriable.h"
#include "QueriableDescriptor.h"
#include "QueryDescriptor.h"
#include "EnumEntry.h"

#include <iostream>
#include <sstream>

Queriable::Queriable()
: _publisherQueriable(false), _queriableName("no name"), _queriableDescription("no description"),
_queriableType("no type")
{
   _qd.setQueriable(this);
}


Queriable::Queriable(const Queriable& q)
   : _queryDescriptor(q._queryDescriptor), 
     _publisherQueriable(q._publisherQueriable),   
     _queriableName (q._queriableName), 
     _queriableDescription(q._queriableDescription),
     _queriableType(q._queriableType)
{
   _qd.setQueriable(this);
   std::unique_ptr<Queriable> dup;
   for (std::list<Queriable*>::const_iterator i = q._queriableList.begin();
	i != q._queriableList.end(); i++) {
      (*i)->duplicate(dup);
      _queriableList.push_back(dup.release());
   }
}


std::list<Queriable*> const & Queriable::getQueriableList() const
{
   return _queriableList;
}


QueriableDescriptor & Queriable::getQueriableDescriptor()
{
   _qd.setName(_queriableName);
   _qd.setDescription(_queriableDescription);
   _qd.setType(_queriableType);
   _qd.setQueriable(this);
   return _qd;
}


QueriableDescriptor & Queriable::getQueriableDescriptor(std::string context)
{
   _qd.setName(_queriableName);
   std::ostringstream ostr;
   ostr<<context<<" : "<<_queriableDescription;
   _qd.setDescription(ostr.str());
   _qd.setType(_queriableType);
   _qd.setQueriable(this);
   return _qd;
}


QueryDescriptor & Queriable::getQueryDescriptor()
{
   return _queryDescriptor;
}


bool Queriable::isPublisherQueriable()
{
   return _publisherQueriable;
}


std::unique_ptr<EnumEntry> & Queriable::emptyEnum()
{
   _aptrEmptyEnum.reset(new EnumEntry("",""));
   return _aptrEmptyEnum;
}


Queriable::~Queriable()
{
   std::list<Queriable*>::iterator iter = _queriableList.begin();
   std::list<Queriable*>::iterator end = _queriableList.end();
   for (; iter != end; ++iter)
      delete (*iter);
}
