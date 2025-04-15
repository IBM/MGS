// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FUNCTOR_H
#define FUNCTOR_H
#include "Copyright.h"

#include <memory>
#include <vector>
#include <string>
class DataItem;
class LensContext;

class Functor {
  public:
  Functor();
  virtual const std::string& getCategory() const;
  void initialize(LensContext* c, const std::vector<DataItem*>& args);
  void execute(LensContext* c, const std::vector<DataItem*>& args,
               std::unique_ptr<DataItem>& rvalue);
  Functor(const Functor& rv) {};
  Functor& operator=(const Functor& rv) {
    return *this;
  };
  virtual void duplicate(std::unique_ptr<Functor>&& fap) const = 0;
  virtual ~Functor();

  protected:
  virtual void doInitialize(LensContext* c,
                            const std::vector<DataItem*>& args) = 0;
  virtual void doExecute(LensContext* c, const std::vector<DataItem*>& args,
                         std::unique_ptr<DataItem>& rvalue) = 0;

  public:
  static const std::string _category;
};
#endif
