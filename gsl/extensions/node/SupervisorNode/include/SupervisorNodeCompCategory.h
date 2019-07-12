#ifndef SupervisorNodeCompCategory_H
#define SupervisorNodeCompCategory_H

#include "Lens.h"
#include "CG_SupervisorNodeCompCategory.h"
#include "mnist/mnist_reader.hpp"

<<<<<<< HEAD
#include <vector>

=======
>>>>>>> Adding DNN model suite.
class NDPairList;

class SupervisorNodeCompCategory : public CG_SupervisorNodeCompCategory
{
   public:
      SupervisorNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void updateShared(RNG& rng);
<<<<<<< HEAD
      
    private:
      mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;
      void outputError(unsigned);
      bool isReady();
      void shuffleDeck(unsigned, RNG& rng);
      ShallowArray<unsigned> _shuffledDeck;
=======

   private:
      mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;
      void outputError();
>>>>>>> Adding DNN model suite.
};

#endif
