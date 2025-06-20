// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
   f->duplicate(std::move(fap));
   _data = fap.release();
}


std::string FunctorDataItem::getString(Error* error) const
{
   return "";
}
