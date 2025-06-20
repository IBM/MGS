// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TRIGGEREDPAUSEACTION_H
#define TRIGGEREDPAUSEACTION_H
#include "Copyright.h"

#include <list>
#include <memory>

class PauseActionable;

class TriggeredPauseAction : public std::list<PauseActionable*>
{
   public:
      TriggeredPauseAction();
      void startAction();
      void insert (std::unique_ptr<PauseActionable>& item);
      ~TriggeredPauseAction();

   private:
      std::list<PauseActionable*> _list;
};
#endif
