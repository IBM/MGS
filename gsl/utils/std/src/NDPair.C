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

#include "NDPair.h"
#include "StringDataItem.h"
#include "IntDataItem.h"
#include "DoubleDataItem.h"

NDPair::NDPair(const std::string& name, std::unique_ptr<DataItem>& di)
: _name(name)
{
   _dataItem = std::move(di);
}

NDPair::NDPair(const std::string& name, const std::string& value)
: _name(name)
{
  _dataItem.reset(new StringDataItem(value));
}

NDPair::NDPair(const std::string& name, int value)
: _name(name)
{
  _dataItem.reset(new IntDataItem(value));
}

NDPair::NDPair(const std::string& name, double value)
: _name(name)
{
  _dataItem.reset(new DoubleDataItem(value));
}


int NDPair::operator==(const std::string& name) const
{
   if(_name == name) {
      return(1);
   }
   else {
      return(0);
   }
}


int NDPair::operator!=(const std::string& name) const
{
   if(_name != name) {
      return(1);
   }
   else {
      return(0);
   }
}

NDPair::NDPair(const NDPair& rv)
{
   copyContents(rv);
}


const NDPair& NDPair::operator=(const NDPair& rv)
{
   if (this == &rv) {
      return *this;
   }
   destructContents();
   copyContents(rv);
   return *this;
}


const std::string& NDPair::getName() const
{
   return(_name);
}


std::string NDPair::getValue() const
{
   if (_dataItem.get()) {
      return _dataItem->getString();
   } 
   return "NoDataItem";
}

void NDPair::setValue(const std::string& value)
{
  _dataItem.reset(new StringDataItem(value));
}

void NDPair::setValue(int value)
{
  _dataItem.reset(new IntDataItem(value));
}

void NDPair::setValue(double value)
{
  _dataItem.reset(new DoubleDataItem(value));
}

DataItem* NDPair::getDataItem() const
{
   return _dataItem.get();
}

void NDPair::setDataItem(DataItem* di)
{
   di->duplicate(_dataItem);
}


void NDPair::getDataItemOwnership(std::unique_ptr<DataItem>& di)
{
   di = std::move(_dataItem);
}

void NDPair::setDataItemOwnership(std::unique_ptr<DataItem>& di)
{
   _dataItem = std::move(di);
}

NDPair::~NDPair()
{
   destructContents();
}

void NDPair::copyContents(const NDPair& rv)
{
   _name = rv._name;
   if (rv._dataItem.get()) {
      rv._dataItem->duplicate(_dataItem);
   }
}

void NDPair::destructContents()
{
}
