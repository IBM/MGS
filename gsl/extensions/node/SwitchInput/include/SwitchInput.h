#ifndef SwitchInput_H
#define SwitchInput_H

#include "Lens.h"
#include "CG_SwitchInput.h"
#include "rndm.h"
#include <fstream>


class SwitchInput : public CG_SwitchInput
{
   public:
      void initialize(RNG& rng);
<<<<<<< HEAD
      //void update(RNG& rng);
=======
      void update(RNG& rng);
>>>>>>> origin/team-A
      void outputDrivInp(std::ofstream &);
      virtual ~SwitchInput();

};

#endif
