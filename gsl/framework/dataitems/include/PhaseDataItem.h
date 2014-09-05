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

#ifndef PHASEDATAITEM_H
#define PHASEDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include <vector>
#include <memory>

class Phase;

class PhaseDataItem : public DataItem
{

   private:
      Phase *_data;
      void copyOwnedHeap(const PhaseDataItem& rv);
      void destructOwnedHeap();

   public:
      static char const* _type;

      PhaseDataItem& operator=(const PhaseDataItem& rv);

      // Constructors
      PhaseDataItem();
      PhaseDataItem(std::auto_ptr<Phase> data);
      PhaseDataItem(const PhaseDataItem& rv);

      // Destructor
      ~PhaseDataItem();

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Singlet Methods
      Phase* getPhase(Error* error=0) const;
      void setPhase(std::auto_ptr<Phase>& data, Error* error=0);
      std::string getString(Error* error=0) const;

};
#endif
