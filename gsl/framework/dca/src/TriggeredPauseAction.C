// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

void TriggeredPauseAction::insert(std::unique_ptr<PauseActionable>& item)
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
