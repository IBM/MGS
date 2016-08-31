#ifndef _GlobalNTSConfig_H
#define _GlobalNTSConfig_H


class GlobalNTS{
	public:
	static const int anybranch_at_end = -1;
  static const float shellDepth ; //[um]
};

const float GlobalNTS::shellDepth = 0.0 ; //[um]
//NOTE: shellDepth <= 0.0 which means it use the default value defined in CaConcentrationJunction
#endif
