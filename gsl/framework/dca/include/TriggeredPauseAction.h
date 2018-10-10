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
