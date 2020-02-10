#ifndef VanDerPolCoupledSystem_H
#define VanDerPolCoupledSystem_H

#include "Lens.h"
#include "CG_VanDerPolCoupledSystem.h"
#include "rndm.h"

#include <fstream>
#include <iomanip>
#include <algorithm>
#include <memory>

/*  A number of things to be added
 *  1. #include "NumSolver_RK4.h"
 *  2. define StateType
 *  3. derive from proper RK4 stepper 
 *  4. RHS operator 
 *  5. define StateType variable
 */
// step 1
#include "NumSolver_RK4.h"

// step 2
typedef std::vector<double> VanDerPolCoupledSystem_StateType; 
typedef double TimeType;
typedef double VarType;
//typedef NumSolver_RK4< VanDerPolCoupledSystem_StateType > RK4_CPU_stepper;
// step 3

class VanDerPolCoupledSystemCompCategory;

class VanDerPolCoupledSystem : public CG_VanDerPolCoupledSystem
               , public NumSolver_RK4< VanDerPolCoupledSystem_StateType, VanDerPolCoupledSystem_StateType, VarType, TimeType >
{
   public:
      void initializeNode(RNG& rng);
      void initializeSolver(RNG& rng);
      // step 4
      void operator() (const VanDerPolCoupledSystem_StateType &x, VanDerPolCoupledSystem_StateType &dxdt, TimeType t);
      void update1(RNG& rng);
      void update2(RNG& rng);
      void update3(RNG& rng);
      void update4(RNG& rng);
      virtual ~VanDerPolCoupledSystem();
   private:
      // step 5
      VanDerPolCoupledSystem_StateType x;
   protected:
      void update(int i);
      void initSolver();
      RNG* _rng;
      /* I/O */
      int ioStride;  // how often (i.e. numIterations) to write data
      int prevIOIteration;  //the iteration at which I/O was done
      int ioCounterFlush; 
      std::shared_ptr<std::ofstream> outFile;
      void open_file(std::string brain_area, std::string neuron_name, int nodeIdx);
      void write_file(bool now=false);
    public:
      void setRealCompCategory(VanDerPolCoupledSystemCompCategory* target);
      VanDerPolCoupledSystemCompCategory* getRealCompCategory(){ return _comCat;};
      VanDerPolCoupledSystemCompCategory* _comCat;
};

#endif
