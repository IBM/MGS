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
