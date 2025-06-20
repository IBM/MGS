// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "SupervisorNode.h"
#include "CG_SupervisorNode.h"
#include "rndm.h"
#include "IsToast.h"
#include <math.h>
#include <algorithm>
#include <cfloat>
#ifdef HAVE_GPU
#include "CG_SupervisorNodeCompCategory.h"
#endif

#ifdef HAVE_GPU

#define primaryGradient  (_container->um_primaryGradient[__index__])
#define predictions  (_container->um_predictions[__index__])
#define logits  (_container->um_logits[__index__])
#define sumOfSquaredError  (_container->um_sumOfSquaredError[__index__])
#define wins  (_container->um_wins[__index__])
#define ready  (_container->um_ready[__index__])
#endif


#define PRELIM_STATE DBL_MAX
#define SHD getSharedMembers()

void SupervisorNode::initialize(RNG& rng) 
{
  /*
   * If a network has N_LABELS choice for prediction, then in softmax-based prediction
   * we ave N_LABELS softmax values
   * As each softmax value is calculated by one SupervisorNode node
   * Each SupervisorNode supervises for one label prediction
   *     softmax -> to calculate the predictions (i.e. prob. value), 
   *     the SupervisorNode node receives the 'output' from all last-layer DNNode(s)
   *     and store in the '*[] logits' array
   *     so all the l
   */
  primaryGradient = PRELIM_STATE;
  predictions.increaseSizeTo(logits.size());
}

void SupervisorNode::update(RNG& rng) 
{
  auto end = logits.end();  
  if (!ready) {
    for (auto iter=logits.begin(); iter!=end; ++iter) {
      ready = (**iter != PRELIM_STATE);
      if (!ready) break;
    }
  }
  if (ready) {
    double sumOfExp=0;
    double maxElement = 0; //original
    //double maxElement = **max_element_dereference<Array_FlatIterator<double*, double*>>(logits.begin(), logits.end(), dereference_compare);
    for (auto iter=logits.begin(); iter!=end; ++iter) 
      sumOfExp+=exp(**iter - maxElement);

    auto prob_iter=predictions.begin(); //softmax

    unsigned h=0;
    unsigned winner=-1; //predicted label
    double maxProbability=-DBL_MAX;
    for (auto iter=logits.begin(); iter!=end; ++iter, ++prob_iter, ++h) {
      *prob_iter = exp(**iter - maxElement)/sumOfExp;
      if (*prob_iter>maxProbability) {
	winner=h;
	maxProbability=*prob_iter;
      }
    }

    if (winner==SHD.labels[SHD.labelIndex]) ++wins;
    
    // NOTE: supervisor of globalIndex 'i' in charge of predicting if input is label 'i' 
    // So the ideal probability of the 'prediction' of the current supervisor
    // a one-hot encoded vector ~ only one element has value 1, all others get 0
    // here is the element i-th in the one-hot vector
    double oneHot = (SHD.labels[SHD.labelIndex] == getGlobalIndex()) ? 1.0 : 0.0;

    auto my_prob_iter = predictions.begin()+getGlobalIndex();

    double error = oneHot - *my_prob_iter;
    if (SHD.refreshErrors) {
      sumOfSquaredError=0;
      wins=0;
    }
    sumOfSquaredError += error * error;

    primaryGradient=0;
    if (!SHD.test) {
      h=0;
      auto prob_iter=predictions.begin(); 
      auto prop_end=predictions.end();
      for (; prob_iter!=prop_end; ++h, ++prob_iter) {
	primaryGradient += *my_prob_iter * 
	  ( ( (prob_iter==my_prob_iter) ? 1.0 : 0.0 ) - *prob_iter ) * 
	  ( ( (SHD.labels[SHD.labelIndex] == h) ? 1.0 : 0.0 ) - *prob_iter );
      }
    }
  }
}

SupervisorNode::~SupervisorNode() 
{
}

