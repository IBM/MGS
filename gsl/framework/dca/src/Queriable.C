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
   std::auto_ptr<Queriable> dup;
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


std::auto_ptr<EnumEntry> & Queriable::emptyEnum()
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
