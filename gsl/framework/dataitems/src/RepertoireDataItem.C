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

#include "RepertoireDataItem.h"
#include <iostream>
#include <sstream>

// Type
const char* RepertoireDataItem::_type = "REPERTOIRE";

// Constructors
RepertoireDataItem::RepertoireDataItem(Repertoire *repertoire) 
   : _repertoire(repertoire)
{
}


std::string RepertoireDataItem::getString(Error* error) const
{
   std::ostringstream os;
   os <<_repertoire;
   return os.str();
}


RepertoireDataItem::RepertoireDataItem(const RepertoireDataItem& DI)
{
   _repertoire = DI._repertoire;
}


// Utility methods
void RepertoireDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new RepertoireDataItem(*this)));
}


RepertoireDataItem& RepertoireDataItem::operator=(const RepertoireDataItem& DI)
{
   setRepertoire(DI.getRepertoire());
   return(*this);
}


const char* RepertoireDataItem::getType() const
{
   return _type;
}


Repertoire* RepertoireDataItem::getRepertoire() const
{
   return _repertoire;
}


void RepertoireDataItem::setRepertoire(Repertoire *rp)
{
   _repertoire = rp;
}
