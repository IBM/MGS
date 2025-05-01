// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Zipper_H
#define Zipper_H

#include "Mgs.h"
#include "CG_ZipperBase.h"
#include "LensContext.h"
#include <memory>
#include <map>

class NoConnectConnector;
class GranuleConnector;
class LensConnector;

class Zipper : public CG_ZipperBase
{
   public:
      void userInitialize(LensContext* CG_c);
      void userExecute(LensContext* CG_c, std::vector<DataItem*>::const_iterator begin, std::vector<DataItem*>::const_iterator end);
      Zipper();
      Zipper(Zipper const &);
      virtual ~Zipper();
      virtual void duplicate(std::unique_ptr<Zipper>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_ZipperBase>&& dup) const;
   private:
      NoConnectConnector* _noConnector;
      GranuleConnector* _granuleConnector;
      LensConnector* _lensConnector;
      //map < name-passed-at-last-argument, vector<giving proportional value along branches being used
      std::map<std::string, std::vector<double> > _branchPropListMap; //keep the list 
};

#endif
