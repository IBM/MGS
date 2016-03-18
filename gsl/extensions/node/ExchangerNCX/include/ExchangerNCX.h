#ifndef ExchangerNCX_H
#define ExchangerNCX_H

#include "Lens.h"
#include "CG_ExchangerNCX.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#ifndef BASED_TEMPERATURE 
#define BASED_TEMPERATURE 35.0 //Celcius
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif
class ExchangerNCX : public CG_ExchangerNCX
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ExchangerNCX();
	 private:
};

#endif
