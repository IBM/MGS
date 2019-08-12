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

#include "FunctorDataItem.h"
#include "Functor.h"

// Type
const char* FunctorDataItem::_type = "FUNCTOR";

// Constructors
FunctorDataItem::FunctorDataItem()
   : _data(0)
{
}

FunctorDataItem::FunctorDataItem(std::unique_ptr<Functor>& data)
{
   _data = data.release();
}

FunctorDataItem::~FunctorDataItem()
{
   delete _data;
}


FunctorDataItem::FunctorDataItem(const FunctorDataItem& DI)
:_data(0)
{
   setFunctor(DI._data);
}


// Utility methods
void FunctorDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new FunctorDataItem(*this));
}


FunctorDataItem& FunctorDataItem::operator=(const FunctorDataItem& DI)
{
   setFunctor(DI.getFunctor());
   return(*this);
}


const char* FunctorDataItem::getType() const
{
   return _type;
}


// Singlet methods

Functor* FunctorDataItem::getFunctor(Error* error) const
{
   return _data;
}


void FunctorDataItem::setFunctor(Functor *f, Error* error)
{
   delete _data;
   std::unique_ptr<Functor> fap;
   f->duplicate(fap);
   _data = fap.release();
}


std::string FunctorDataItem::getString(Error* error) const
{
   return "";
}
