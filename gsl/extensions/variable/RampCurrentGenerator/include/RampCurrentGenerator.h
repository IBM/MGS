// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef RampCurrentGenerator_H
#define RampCurrentGenerator_H

#include "Mgs.h"
#include "CG_RampCurrentGenerator.h"
#include <memory>
#include <fstream>

class RampCurrentGenerator : public CG_RampCurrentGenerator
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void finalize(RNG& rng);
      RampCurrentGenerator();
      virtual ~RampCurrentGenerator();
      virtual void duplicate(std::unique_ptr<RampCurrentGenerator>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_RampCurrentGenerator>&& dup) const;
   private:
      void update_RampProtocol(RNG& , float currentTime);
      float tstart, tend; //[ms]
      float nextPulse; //[ms]
      bool first_enter_pulse;
      std::ofstream* outFile = 0;
      float time_write_data; // [ms]
      void (RampCurrentGenerator::*fpt_update)(RNG& rng, float currentTime) = NULL;
      void dataCollection(float currentTime);
      int n_timepoints;
      int current_index; //keep track of index in time_points for the current period
      float Istart;  // [pA]
      float Iend;  // [pA]
      float duration;
};

#endif
