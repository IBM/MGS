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

#include "TriggeredPauseAction.h"
#include "PauseActionable.h"

TriggeredPauseAction::TriggeredPauseAction()
{
}


void TriggeredPauseAction::startAction()
{
   TriggeredPauseAction::iterator end = _list.end();
   for (TriggeredPauseAction::iterator iter = _list.begin(); iter != end; ++iter) {
      (*iter)->action();
   }
}

void TriggeredPauseAction::insert(std::auto_ptr<PauseActionable>& item)
{
   _list.push_back(item.release());
}

TriggeredPauseAction::~TriggeredPauseAction()
{
   TriggeredPauseAction::iterator end = _list.end();
   for (TriggeredPauseAction::iterator iter = _list.begin(); iter != end; ++iter) {
      delete (*iter);
   }   
}
