#ifndef CG_SUPERVISORNODECOMPCATEGORY_CU
#define CG_SUPERVISORNODECOMPCATEGORY_CU

#define PRELIM_STATE DBL_MAX

void __global__ SupervisorNode_kernel_update(
   double* primaryGradient
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double*, Array_Flat<int>::MemLocation::UNIFIED_MEM>* logits
   #endif
   #if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
   , ShallowArray_Flat<double, Array_Flat<int>::MemLocation::UNIFIED_MEM>* predictions
   #endif
   , unsigned* globalIdx
   , double* sumOfSquaredError
   , unsigned* wins
   , bool* ready
   , unsigned size
   , bool refreshErrors
   , bool test
   , unsigned labelIndex
   , unsigned* labels
)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   if (index < size) {
      // add your code here
      auto end = logits[index].end();  
      if (!ready[index]) {
         for (auto iter=logits[index].begin(); iter!=end; ++iter) {
            ready[index] = (**iter != PRELIM_STATE);
            if (!ready[index]) break;
         }
      }
      if (ready[index]) {
         double sumOfExp=0;
         double maxElement = 0; //original
         //double maxElement = **max_element_dereference<Array_FlatIterator<double*, double*>>(logits.begin(), logits.end(), dereference_compare);
         for (auto iter=logits[index].begin(); iter!=end; ++iter) 
            sumOfExp+=exp(**iter - maxElement);

         auto prop_iter=predictions[index].begin();

         unsigned h=0;
         /* TUAN TODO: double check why unsigned and assign negative value */
         unsigned winner=-1;
         double maxProbability=-DBL_MAX;
         for (auto iter=logits[index].begin(); iter!=end; ++iter, ++prop_iter, ++h) {
            *prop_iter = exp(**iter - maxElement)/sumOfExp;
            if (*prop_iter>maxProbability) {
               winner=h;
               maxProbability=*prop_iter;
            }
         }
         int label = labels[labelIndex];
         if (winner==label) ++wins[index];

         //double oneHot = (labels[labelIndex] == getGlobalIndex()) ? 1.0 : 0.0;
         double oneHot = (label == globalIdx[index]) ? 1.0 : 0.0;

         /* TODO -> replace getGlobalIndex() */
         //auto my_prop_iter = predictions[index].begin()+getGlobalIndex();
         auto my_prop_iter = predictions[index].begin()+globalIdx[index];

         double error = oneHot - *my_prop_iter;
         if (refreshErrors) {
            sumOfSquaredError[index]=0;
            wins[index]=0;
         }
         sumOfSquaredError[index] += error * error;

         primaryGradient[index]=0;
         if (!test) {
            h=0;
            auto prop_iter=predictions[index].begin(); 
            auto prop_end=predictions[index].end();
            for (; prop_iter!=prop_end; ++h, ++prop_iter) {
               primaryGradient[index] += *my_prop_iter * 
                  ( ( (prop_iter==my_prop_iter) ? 1.0 : 0.0 ) - *prop_iter ) *
                  ( ( (label == h) ? 1.0 : 0.0 ) - *prop_iter );
            }
         }
      }
   }
}
#endif
