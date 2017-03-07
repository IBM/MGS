#include <math.h>
#include <pthread.h>
#include <algorithm>
#include <iostream>
#include <vector>
int main()
{
 #define dyn_var_t double
#define LOOKUP_TAUM_LENGTH 16  // size of the below array
  const dyn_var_t _Vmrange_taum[] = {-100, -90, -80, -70, -60, -50, -40, -30,
                                   -20,  -10, 0,   10,  20,  30,  40,  50};
  static std::vector<dyn_var_t> Vmrange_taum;
  std::vector<dyn_var_t> tmp(_Vmrange_taum,
      _Vmrange_taum + LOOKUP_TAUM_LENGTH);
  Vmrange_taum = tmp;
  float v = 60;
    std::vector<dyn_var_t>::iterator low =
        std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
    int index = low - Vmrange_taum.begin();
    std::cout << index << " " << Vmrange_taum[15] <<  " " << Vmrange_taum[index] <<  std::endl; 
    return 0;
}
