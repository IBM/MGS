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

#ifndef C_PHASE_LIST_H
#define C_PHASE_LIST_H
#include "Copyright.h"

#include <vector>
#include <memory>

#include "C_production.h"

class LensContext;
class DataItem;
class C_phase;

class C_phase_list : public C_production
{
   public:
      C_phase_list(const C_phase_list&);
      C_phase_list(C_phase *, SyntaxError *);
      C_phase_list(C_phase_list *, C_phase *, SyntaxError *);
      void releaseList(std::auto_ptr<std::vector<C_phase*> >& phases);
      virtual ~C_phase_list();
      virtual C_phase_list* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      std::vector<C_phase*>* _phases;
};
#endif
