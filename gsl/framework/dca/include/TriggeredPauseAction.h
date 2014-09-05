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
      void insert (std::auto_ptr<PauseActionable>& item);
      ~TriggeredPauseAction();

   private:
      std::list<PauseActionable*> _list;
};
#endif
