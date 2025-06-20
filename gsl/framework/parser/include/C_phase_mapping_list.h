// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_PHASE_MAPPING_LIST_H
#define C_PHASE_MAPPING_LIST_H
#include "Copyright.h"

#include <vector>
#include <memory>

#include "C_production.h"

class GslContext;
class DataItem;
class C_phase_mapping;

class C_phase_mapping_list : public C_production
{
   public:
      C_phase_mapping_list(const C_phase_mapping_list&);
      C_phase_mapping_list(C_phase_mapping *, SyntaxError *);
      C_phase_mapping_list(C_phase_mapping_list *, C_phase_mapping *, 
			    SyntaxError *);
      void releaseList(std::unique_ptr<std::vector<C_phase_mapping*> >& phases);
      virtual ~C_phase_mapping_list ();
      virtual C_phase_mapping_list* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      std::vector<C_phase_mapping*>* _phase_mappings;
};
#endif
