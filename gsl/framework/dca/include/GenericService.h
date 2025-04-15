// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GenericService_H
#define GenericService_H
#include "Copyright.h"

#include "Service.h"
#include "TypeClassifier.h"
#include "Publishable.h"

#include <string>
#include <vector>
#include <memory>
#include <sstream>

class DataItem;
class Publishable;

template <class T>
class GenericService : public Service
{

  public:
  GenericService(Publishable* publishable, T* data)
      : _publishable(publishable), _data(data)
  {
  }
  virtual std::string getName() const
  {
    return _publishable->getServiceName(_data);
  }
  virtual std::string getDescription() const
  {
    return _publishable->getServiceDescription(_data);
  }
  virtual std::string getDataItemDescription() const
  {
    return TypeClassifier<T>::getName();
  }

  virtual std::string getStringValue() const
  {
    std::ostringstream os;
    os << *_data;
    return os.str();
  }
  virtual void setStringValue(const std::string& value)
  {
    std::istringstream is(value);
    is >> *_data;
  }
  T* getData() { return _data; }

  virtual void duplicate(std::unique_ptr<Service>& dup) const
  {
    dup.reset(new GenericService<T>(*this));
  }
  virtual ~GenericService() {}

  private:
  Publishable* _publishable;
  T* _data;
};
#endif
