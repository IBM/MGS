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

#ifndef C_PHASE_MAPPING_LIST_H
#define C_PHASE_MAPPING_LIST_H
#include "Copyright.h"

#include <vector>
#include <memory>

#include "C_production.h"

class LensContext;
class DataItem;
class C_phase_mapping;

class C_phase_mapping_list : public C_production
{
   public:
      C_phase_mapping_list(const C_phase_mapping_list&);
      C_phase_mapping_list(C_phase_mapping *, SyntaxError *);
      C_phase_mapping_list(C_phase_mapping_list *, C_phase_mapping *, 
			    SyntaxError *);
      void releaseList(std::auto_ptr<std::vector<C_phase_mapping*> >& phases);
      virtual ~C_phase_mapping_list ();
      virtual C_phase_mapping_list* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      std::vector<C_phase_mapping*>* _phase_mappings;
};
#endif
