#include <algorithm>
#include <iostream>
#include <math.h>
#include <pthread.h>
#include <vector>
#include <assert.h>
template<typename T>
T linear_interp( T x0, T y0, T x1, T y1, T x )
{
  T a = (y1-y0)/(x1-x0);//tan(alpha)
  T b = -a*x0+y0;
  T y = a * x + b;
  return y;
}
int main()
{
#define dyn_var_t double
#define LOOKUP_TAUM_LENGTH 16  // size of the below array
  const dyn_var_t _Vmrange_taum[] = {-100, -90, -80, -70, -60, -50, -40, -30,
                                     -20,  -10, 0,   10,  20,  30,  40,  50};
  static std::vector<dyn_var_t> Vmrange_taum;
  std::vector<dyn_var_t> tmp(_Vmrange_taum, _Vmrange_taum + LOOKUP_TAUM_LENGTH);
  Vmrange_taum = tmp;
  dyn_var_t taumNat[] = {0.02, 0.06, 0.07, 0.09, 0.11, 0.13, 0.20, 0.32,
                       0.16, 0.15, 0.12, 0.08, 0.06, 0.06, 0.06, 0.06};
  for (std::vector<dyn_var_t>::const_iterator i = Vmrange_taum.begin(); i != Vmrange_taum.end(); ++i)
    std::cout << *i << ' ';
  std::cout << std::endl;

  dyn_var_t v;
  std::vector<dyn_var_t>::iterator low;
  int index;
  dyn_var_t taum;

  v = -100;
  std::cout << "Using v = " << v << std::endl;
  low = std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
  index = low - Vmrange_taum.begin();
  std::cout << index << " " << Vmrange_taum[0] << " " << Vmrange_taum[index]
            << std::endl;
  assert(index == 0);
  assert(Vmrange_taum[index] == Vmrange_taum[0]);
  assert(taumNat[index] == 0.02);

  v = 50;
  low =  std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
  index = low - Vmrange_taum.begin();
  std::cout << "Using v = " << v << std::endl;
  std::cout << index << " " << Vmrange_taum[15] << " " << Vmrange_taum[index]
            << std::endl;
  assert(Vmrange_taum[index] == Vmrange_taum[15]);
  assert(index == LOOKUP_TAUM_LENGTH-1);
  assert(taumNat[index] == 0.06);

  v = 60; // upper the range
  low =  std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
  index = low - Vmrange_taum.begin();
  std::cout << "Using v = " << v << std::endl;
  std::cout << index << " " << Vmrange_taum[LOOKUP_TAUM_LENGTH-1] << " " << Vmrange_taum[index-1]
            << std::endl;
  if (index == 0)
    taum = taumNat[0];
  else if (index < LOOKUP_TAUM_LENGTH)
    taum = linear_interp(Vmrange_taum[index-1], taumNat[index-1], 
        Vmrange_taum[index], taumNat[index], v);
  else //assume saturation in taum when Vm > max-value
    taum = taumNat[index-1];
  assert(index == LOOKUP_TAUM_LENGTH);
  assert(Vmrange_taum[index-1] == Vmrange_taum[LOOKUP_TAUM_LENGTH-1]);
  assert(taumNat[index-1] == 0.06);


  v = -73;
  std::cout << "Using v = " << v << std::endl;
  low = std::lower_bound(Vmrange_taum.begin(), Vmrange_taum.end(), v);
  index = low - Vmrange_taum.begin();
  std::cout << index << " " << Vmrange_taum[15] << " " << Vmrange_taum[index]
            << std::endl;
  if (index == 0)
    taum = taumNat[0];
  else if (index < LOOKUP_TAUM_LENGTH)
    taum = linear_interp(Vmrange_taum[index-1], taumNat[index-1], 
        Vmrange_taum[index], taumNat[index], v);
  else //assume saturation in taum when Vm > max-value
    taum = taumNat[index-1];

  std::cout << "taum = " << taum << std::endl;
  // assert(taumNat[index] = 0.02);

  return 0;
}
