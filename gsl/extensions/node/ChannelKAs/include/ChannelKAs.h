#ifndef ChannelKAs_H
#define ChannelKAs_H

#include "Lens.h"
#include "CG_ChannelKAs.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream>

#if CHANNEL_KAs == KAs_WOLF_2005
#define BASED_TEMPERATURE 15.0  // Celcius
#define Q10 3.0

#elif CHANNEL_KAs == KAs_KORNGREEN_SAKMANN_2000
#define BASED_TEMPERATURE 21.0  // Celcius
#define Q10 2.3

#elif CHANNEL_KAs == KAs_MAHON_2000        
#define BASED_TEMPERATURE 22.0  // Celcius 
#define Q10 2.5                            
#endif

#ifndef Q10
#define Q10 3.0  // default
#endif
class ChannelKAs : public CG_ChannelKAs
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~ChannelKAs();
  static void initialize_others();  // new
  private:
#if defined(WRITE_GATES)      
  std::ofstream* outFile;     
  float _prevTime;            
  static SegmentDescriptor _segmentDescriptor;
#define IO_INTERVAL 0.1 // ms 
#endif                        
};

#endif
