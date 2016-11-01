from genSpines import SomeClass
targetDir = "neurons"
neuron = SomeClass(targetDir +"/neuron.swc")
##########################
### APICAL DEN (apical/tufted)
#dfrom = 915.0 # um
#dto = 925.0 # um
dfrom = 755.0 # um
dto = 765.0 # um
#dfrom = 615.0 # um
#dto = 625.0 # um
#dfrom = 315.0 # um
#dfrom = 385.0 # um
#dto = 395.0 # um
#dto = 325.0 # um
#dfrom = 115.0 # um
#dto = 125.0 # um
#neuron.checkSiteApical(dfrom, dto )
#dfrom = 45.0 # um
#dto = 75.0 # um
#neuron.checkSiteApical(dfrom, dto )

### Get the list of point in the morphology falling within the given range
#neuron.checkSiteApical()


##########################
### APICAL DEN (apical/tufted)
dfrom = 115.0 # um
dto = 125.0 # um
#neuron.checkSiteBasal(dfrom, dto )

##########################
### AXON/AIS
#dfrom = 35.0 # um
#dto = 45.0 # um
dfrom = 40.0 # um
dto = 55.0 # um
#dfrom = 50.0 # um
#dto = 65.0 # um
neuron.checkSiteAxon(dfrom, dto )

#############################
## site = (x,y,z,r)
#site = [82.727, -68.87, 18.563, 2.0]
#site = [1.23, 20.415, 6.3015, 5.0]
#site = [1.4, 332.0, -6.68, 5.0]
#site = [68.9, 679.55, 21.87, 5.0]
#site = [64.13, 834.2, -14.87, 5.0]
#site = [28, 679, -16, 5.0]
#site = [-41, 233, 6, 3.0]
#site = [-88, 74, -1, 3.0]
#neuron.getDist2Soma(site)

