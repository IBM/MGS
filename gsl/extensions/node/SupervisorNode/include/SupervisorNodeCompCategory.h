// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef SupervisorNodeCompCategory_H
#define SupervisorNodeCompCategory_H

#include "Mgs.h"
#include "CG_SupervisorNodeCompCategory.h"
#include "mnist/mnist_reader.hpp"

#include <vector>

class NDPairList;

class SupervisorNodeCompCategory : public CG_SupervisorNodeCompCategory
{
   public:
      SupervisorNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void initializeShared(RNG& rng);
      void updateShared(RNG& rng);
#ifdef HAVE_GPU
      void loadDataToGPU();
      void updateShared_GPU(RNG& rng);
#endif
      void updateShared_origin(RNG& rng);

      ~SupervisorNodeCompCategory();
      
    private:
      /* MNIST_dataset<Container, Image, Label> */
      /* with fields:
         .training_images    Container<Image>
         .training_labels    Container<Label>
         .test_images        Container<Image>
         .test_labels
         */
      mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;
      void outputError(unsigned);
      bool isReady();
      void shuffleDeck(unsigned, RNG& rng);
      //store the indices of image to be passed into DNN
      //in that order
      ShallowArray<unsigned> _shuffledDeck;
      unsigned count_train;
      unsigned count_test;
      //DEBUG purpose
      bool loaded_train_image = false;
      bool loaded_test_image = false;
};

#endif
