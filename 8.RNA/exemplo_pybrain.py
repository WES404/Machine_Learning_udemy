from pybrain.structure import FeedFowardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConmection

rede = FeedFowardNetwork()

camadaEntrada = LinearLayer(2)
camadaOculta = SigmoidLayer(3)
camadaSaida =  SigmoidLayer(1)
bias1 = BiasUnit()
bias2 = BiasUnit()

rede.addmodule(camadaEntrada)
rede.addmodule(camadaOculta)
rede.addmodule(camadaSaida)
rede.addmodule(bias1)
rede.addmodule(bias2)

entradaOculta = FullConection(camadaEntrada, camadaOculta)
ocultaSaida = FullConmection(camadaOculta, camadaSaida)
biasOculta = FullConmection(bias1, camadaOculta)
biasSaida = FullConmection(bias2, camadaSaida)

rede.sortModules()