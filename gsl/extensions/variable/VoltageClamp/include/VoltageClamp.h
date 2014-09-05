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

#ifndef VoltageClamp_H
#define VoltageClamp_H

#include "Lens.h"
#include "CG_VoltageClamp.h"
#include <fstream>
#include <memory>

class VoltageClamp : public CG_VoltageClamp
{
   public:
      void initialize(RNG& rng);
      void updateI(RNG& rng);
      void finalize(RNG& rng);
      virtual void startWaveform(Trigger* trigger, NDPairList* ndPairList);
      virtual void setCommand(Trigger* trigger, NDPairList* ndPairList);
      virtual void toggle(Trigger* trigger, NDPairList* ndPairList);
      VoltageClamp();
      virtual ~VoltageClamp();
      virtual void duplicate(std::auto_ptr<VoltageClamp>& dup) const;
      virtual void duplicate(std::auto_ptr<Variable>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_VoltageClamp>& dup) const;
   private:
      std::ofstream* outFile;
};

#endif
