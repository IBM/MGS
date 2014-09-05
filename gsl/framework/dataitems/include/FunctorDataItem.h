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

#ifndef FUNCTORDATAITEM_H
#define FUNCTORDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include <vector>

class Functor;

class FunctorDataItem : public DataItem
{

   private:
      Functor *_data;

   public:
      static char const* _type;

      FunctorDataItem & operator=(const FunctorDataItem &);

      // Constructors
      FunctorDataItem();
      FunctorDataItem(std::auto_ptr<Functor> data);
      FunctorDataItem(const FunctorDataItem& DI);

      // Destructor
      ~FunctorDataItem();

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Singlet Methods
      Functor* getFunctor(Error* error=0) const;
      void setFunctor(Functor *, Error* error=0);
      std::string getString(Error* error=0) const;

};
#endif
