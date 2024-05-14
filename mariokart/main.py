import collections
import csv

import random
import operator
import math
import itertools
import math
from random import Random
from time import time
from inspyred import ec, benchmarks

import numpy

from functools import partial





#Fórmulas:
# Tiempo en Recorrer una distancia: Cambio en el tiempo = Metros Recorridos / Velocidad
# Tiempo en Acelerar o Decelerar a una velocidad:Cambio en el tiempo = Diferencia de Velocidad / (aceleración * (Parámetro*Peso))
# Distancia Recorrida para pasar de una velocidad a otra:  VF^2 = VI^2 + 2*a*d
#   VF = Velocidad Final
#   VI = Velocidad Inicial
#   a = Aceleración
#   d = distancia recorrida
# Velocidad Máxima en curva:
#  Curva Abierta = 0.75 de la Velocidad Maxima * (Traccion/10)
#  Curva Media = 0.5 de la Velocidad Maxima * (Traccion/10)
#  Curva Cerrada = 0.25 de la Velocidad Maxima * (Traccion/10)
#  MetrosRectaMaximaVelocidad = MetrosRectaTotal - MetrosAcelerar(Distancia Recorrida para pasar de una velocidad a otra) - MetrosFrenar(Distancia Recorrida para pasar de una velocidad a otra)
#  TiempoRectaMaximaVelocidad = MetrosRectaMaximaVelocidad / VelocidadMaxima
#  TiempoAcelerar-TiempoFrenar = Distancia Recorrida para pasar de una velocidad a otra/ (aceleracion-peso)
#









class Coche:
    def __init__(self, chasis, ruedas, parapente, personaje):
        self.chasis = chasis
        self.ruedas = ruedas
        self.parapente = parapente
        self.personaje = personaje
        self.peso = chasis.peso + ruedas.peso + parapente.peso + personaje.peso
        self.aceleracion = chasis.aceleracion + ruedas.aceleracion + parapente.aceleracion + personaje.aceleracion
        self.traccion = chasis.traccion + ruedas.traccion + parapente.traccion + personaje.traccion
        self.miniturbo = chasis.traccion + ruedas.traccion + parapente.traccion + personaje.traccion
        self.velTierra = chasis.velTierra + ruedas.velTierra + parapente.velTierra + personaje.velTierra
        self.velAgua = chasis.velAgua + ruedas.velAgua + parapente.velAgua + personaje.velAgua
        self.velAntiGravedad = chasis.velAntiGravedad + ruedas.velAntiGravedad + parapente.velAntiGravedad + personaje.velAntiGravedad
        self.velAire = chasis.velAire + ruedas.velAire + parapente.velAire + personaje.velAire


    def printCoche(self):
        print("Chasis: " + self.chasis.nombre + " Ruedas: " + self.ruedas.nombre + " Parapente: " + self.parapente.nombre + " Personaje: " + self.personaje.nombre + "Aceleracion: " + str(self.aceleracion) +"Traccion: " + str(self.traccion) + "Peso: " + str(self.peso))

    def calcularTiempoVuelta(self, circuito):
        tiempo = 0
        vActual = 0
        i = 0
        for tramo in circuito:
            if tramo.tipo == "recta":
                if tramo.terreno == "tierra":
                    tiempo += self.calcularTiempoRecta(int(vActual), self.velTierra, calcularVMax(circuito[i+1], self), tramo.longitud,
                                                  self.aceleracion, self.peso, self.velTierra, self.traccion)
                elif tramo.terreno == "agua":
                    tiempo += self.calcularTiempoRecta(int(vActual), self.velAgua, calcularVMax(circuito[i+1], self), tramo.longitud,
                                                  self.aceleracion, self.peso, self.velTierra, self.traccion)
                elif tramo.terreno == "aire":
                    tiempo += self.calcularTiempoRecta(int(vActual), self.velAire, calcularVMax(circuito[i+1], self), tramo.longitud,
                                                  self.aceleracion, self.peso, self.velTierra, self.traccion)
                else:
                    tiempo += self.calcularTiempoRecta(int(vActual), self.velAntiGravedad, calcularVMax(circuito[i+1], self),
                                                  tramo.longitud, self.aceleracion, self.peso, self.velTierra,
                                                  self.traccion)

            else:
                tiempo += calcularTiempoCurva(tramo.longitud, calcularVMax(tramo, self), self.peso, self.traccion, self.velTierra)
                vActual = calcularVMax(circuito[i+1], self) + self.miniturbo * 0.9
                # parametrizar impacto del miniturbo

             return tiempo

     def calcularTiempoRecta(vInicial, vMax, vCurva, mRecta, aceleracion, peso, velocidad, traccion):
            tiempoRecta = 0
            vCurva = vCurva + float(peso) * 0.25 + float(traccion) * 0.75
            mFrenar = distanciaRecorridaAcelerandoFrenando(vInicial,vMax, aceleracion, peso)
            mAcelerar = distanciaRecorridaAcelerandoFrenando(vInicial, vMax, aceleracion, peso)
            mVMax = mRecta - mFrenar
            mAcelerar = calcularMetros(vMax, vInicial, aceleracion, peso)
            if (mAcelerar < mVMax):
                mVMax = mRecta - mAcelerar
                tiempoRecta = calcularTiempo(vMax, vInicial, aceleracion, peso) + calcularTiempoVMax(mVMax,
                                                                                                     velocidad) + calcularTiempo(
                    vCurva, vMax, aceleracion, peso)
            else:
                # en este caso no se alcanza vMax, necesitamos otro método para calcular tiempo acelerando
                tiempoRecta = calcularTiempo(vMax, vInicial, aceleracion, peso) + calcularTiempo(vCurva, vMax,
                                                                                                 aceleracion, peso)

            return tiempoRecta
class Tramo:

    def __init__(self, longitud, terreno, tipo):
        self.longitud = longitud
        self.terreno = terreno
        self.tipo = tipo


class Pieza:
    def __init__(self, nombre, peso , aceleracion , traccion, miniturbo , velTierra , velAgua, velAntiGravedad, velAire):
        self.nombre = nombre
        self.peso = peso
        self.aceleracion = aceleracion
        self.traccion = traccion
        self.miniturbo = miniturbo
        self.velTierra = velTierra
        self.velAgua = velAgua
        self.velAntiGravedad = velAntiGravedad
        self.velAire = velAire

class Chasis(Pieza):
    def __init__(self, nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire):
        super().__init__(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
class Rueda(Pieza):
    def __init__(self, nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire):
        super().__init__(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
class Parapente(Pieza):
    def __init__(self, nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire):
        super().__init__(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
class Personaje(Pieza):
    def __init__(self, nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire):
        super().__init__(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)

#los numeros en azul son parámetros que habrá que cambiar






'''
def calcularTiempoRecta(circuito):
    tiempoRecta = 0
    velocidadInicial = 0
    tiempoAcelerar = 0
    tiempoFrenar = 0
    if(not esInicioCircuito(circuito)):
        velocidadInicial = velocidadMaximaCurva(tramo) + miniturbo
    tiempoAcelerar = (velocidadFinal - velocidadInicial)/(aceleracion)
    tiempoFrenar = (velocidadInicial - velocidadFinal)/(aceleracion)
    tiempoRecta = distanciaRecorridaAcelerandoFrenando()
'''
def distanciaRecorridaAcelerandoFrenando(velocidadInicial, velocidadFinal, aceleracion, peso):
    if velocidadInicial > velocidadFinal:
        distancia = ((velocidadInicial**2 - velocidadFinal**2 )*peso)/(2*aceleracion)
    else:
        distancia = ((velocidadFinal**2 - velocidadInicial**2)*peso)/(2*aceleracion)

    return distancia
'''
def esInicioCircuito(circuito):
    return True
'''
def calcularVMax(tramo, individuo):
        if tramo.tipo == "recta":
            if tramo.terreno == "tierra":
                return individuo.velTierra
            elif tramo.terreno == "agua":
                return individuo.velAgua
            elif tramo.terreno == "aire":
                return individuo.velAire
            else:
                return individuo.velAntiGravedad
        elif tramo.tipo == "curva cerrada":
            return float(50) + float(individuo.peso) * 0.25 + float(individuo.traccion) * 0.75
        elif tramo.tipo == "curva media":
            return 70.0 + float(individuo.peso) * 0.25 + float(individuo.traccion) * 0.75
        else:
            return 90.0 + float(individuo.peso) * 0.25 + float(individuo.traccion) * 0.75

def calcularTiempoCurva(mCurva, vCurva, peso, traccion, velocidad):
    if(velocidad<vCurva):
        vCurva = velocidad
    return mCurva * vCurva

def calcularTiempo(vMax, vInicial, aceleracion, peso):
    return 1

def calcularTiempoVMax(mVMax, velocidad):
    return mVMax*velocidad*2

def calcularMetros(vFinal, vInicial, aceleracion, peso):
    tiempo =calcularTiempo(vFinal, vInicial, aceleracion, peso)
    return vInicial*tiempo+0.5*aceleracion*10*tiempo*tiempo

gliders = []

with open('gliders.csv', "r") as csvfile:
    csv_reader=csv.reader(csvfile, delimiter=";")
    next(csv_reader)
    for row in csv_reader:
        nombre = row[0]
        peso = int(row[1])
        aceleracion = int(row[2])
        traccion = int(row[3])
        miniturbo = int(row[5])
        velTierra = int(row[6])
        velAgua = int(row[7])
        velAntiGravedad = int(row[8])
        velAire = int(row[9])
        glider = Parapente(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
        gliders.append(glider)

for glider in gliders:
    print(f"Nombre: {glider.nombre}, Peso: {glider.peso}, Aceleracion: {glider.aceleracion}, Traccion: {glider.traccion}, Miniturbo: {glider.miniturbo}, Velocidad en asfalto: {glider.velTierra}, Velocidad  en aire:  {glider.velAire}, Velocidad  en antigravedad:  {glider.velAntiGravedad}, Velocidad en agua: {glider.velAgua}")

tires = []

with open('tires.csv', "r") as csvTires:
    csv_reader = csv.reader(csvTires, delimiter=";")
    next(csv_reader)
    for row in csv_reader:
        nombre = row[0]
        peso = int(row[1])
        aceleracion = int(row[2])
        traccion = int(row[3])
        miniturbo = int(row[5])
        velTierra = int(row[6])
        velAgua = int(row[7])
        velAntiGravedad = int(row[8])
        velAire = int(row[9])
        tire = Rueda(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
        tires.append(tire)

for tire in tires:
    print(f"Nombre: {tire.nombre}, Peso: {tire.peso}, Aceleracion: {tire.aceleracion}, Traccion: {tire.traccion}, Miniturbo: {tire.miniturbo}, Velocidad en asfalto: {tire.velTierra}, Velocidad en agua: {tire.velAgua}, Velocidad en antigravedad: {tire.velAntiGravedad}, Velocidad en aire: {tire.velAire}")

drivers = []

with open('drivers.csv', "r") as csvDrivers:
    csv_reader = csv.reader(csvDrivers, delimiter=";")
    next(csv_reader)
    for row in csv_reader:
        nombre = row[0]
        peso = int(row[1])
        aceleracion = int(row[2])
        traccion = int(row[3])
        miniturbo = int(row[5])
        velTierra = int(row[6])
        velAgua = int(row[7])
        velAntiGravedad = int(row[8])
        velAire = int(row[9])
        driver = Personaje(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
        drivers.append(driver)

for driver in drivers:
    print(f"Nombre: {driver.nombre}, Peso: {driver.peso}, Aceleracion: {driver.aceleracion}, Traccion: {driver.traccion}, Miniturbo: {driver.miniturbo}, Velocidad en asfalto: {driver.velTierra}, Velocidad en agua: {driver.velAgua}, Velocidad en antigravedad: {driver.velAntiGravedad}, Velocidad en aire: {driver.velAire}")

bodies = []

with open('bodies_karts.csv', "r") as csvBodies:
    csv_reader = csv.reader(csvBodies, delimiter=";")
    next(csv_reader)
    for row in csv_reader:
        nombre = row[0]
        peso = int(row[1])
        aceleracion = int(row[2])
        traccion = int(row[3])
        miniturbo = int(row[5])
        velTierra = int(row[6])
        velAgua = int(row[7])
        velAntiGravedad = int(row[8])
        velAire = int(row[9])
        bodie = Chasis(nombre, peso, aceleracion, traccion, miniturbo, velTierra, velAgua, velAntiGravedad, velAire)
        bodies.append(bodie)

for bodie in bodies:
    print(f"Nombre: {bodie.nombre}, Peso: {bodie.peso}, Aceleracion: {bodie.aceleracion}, Traccion: {bodie.traccion}, Miniturbo: {bodie.miniturbo}, Velocidad en asfalto: {bodie.velTierra}, Velocidad en agua: {bodie.velAgua}, Velocidad en antigravedad: {bodie.velAntiGravedad}, Velocidad en aire: {bodie.velAire}")

def generarCoche(random):
    glider = random.randint(0, 13)
    tire = random.randint(0,20)
    driver = random.randint(0, 42)
    body = random.randint(0, 39)

    return [body, tire, glider, driver]
def ArrayToCoche(car):
    print(car)
    return Coche(bodies[car[0]],tires[car[1]],gliders[car[2]],drivers[car[3]])

'''
def generarPoblacionInicial(size):
    poblacionInicial=[]
    for i in range(size):
        poblacionInicial.append(generarCoche())
    return poblacionInicial
'''

recta1 = Tramo(100, "asfalto", "recta")
curva1 = Tramo(150, "asfalto", "curva cerrada")
recta2 = Tramo(100, "asfalto", "recta")
curva2 = Tramo(150, "asfalto", "curva cerrada")
ovalo = [recta1, curva1, recta2, curva2]

class DiscreteBounderV2(object):
    """Defines a basic bounding function for numeric lists of discrete values.

    This callable class acts as a function that bounds a
    numeric list to a set of legitimate values. It does this by
    resolving a given candidate value to the nearest legitimate
    value that can be attained. In the event that a candidate value
    is the same distance to multiple legitimate values, the legitimate
    value appearing earliest in the list will be used.

    For instance, if ``[1, 4, 8, 16]`` was used as the *values* parameter,
    then the candidate ``[6, 10, 13, 3, 4, 0, 1, 12, 2]`` would be
    bounded to ``[4, 8, 16, 4, 4, 1, 1, 8, 1]``.

    Public Attributes:

    - *values* -- the set of attainable values
    - *lower_bound* -- the smallest attainable value
    - *upper_bound* -- the largest attainable value

    """
    def __init__(self, lower_bound,upper_bound):
        self.values = [i for i in range (max(upper_bound))]
        print(self.values)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, candidate, args):

        closest = lambda target: min(self.values, key=lambda x: abs(x-target))
        bounded_candidate = candidate
        for i, c in enumerate(bounded_candidate):
            if bounded_candidate[i] < self.lower_bound[i]:
                bounded_candidate[i] = self.lower_bound[i]
            if bounded_candidate[i] > self.upper_bound[i]:
                bounded_candidate[i] = self.upper_bound[i]


        return bounded_candidate


class MarioKart(benchmarks.Benchmark):
    """Defines the Mario Kart benchmark problem.

    Public Attributes:

    - *circuito* -- el circuito para el que optimizar el coche


    """
    def __init__(self, circuito):
        benchmarks.Benchmark.__init__(self, 4)
        self.circuito = circuito
        #max_count = [self.capacity // item[0] for item in self.items]
        self.bounder = DiscreteBounderV2([0,0,0,0],[39,20,13,42])
        self.maximize = False

    def generator(self, random, args):
        """Return a candidate solution for an evolutionary algorithm."""
        return generarCoche(random)

    def evaluator(self, candidates, args):
        """Return the fitness values for the given candidates."""
        fitness = []
        for candidate in candidates:
            ArrayToCoche(candidate).printCoche()
            tiempo = ArrayToCoche(candidate).calcularTiempoVuelta(self.circuito)
            fitness.append(tiempo)
        return fitness

size = 50


problem = MarioKart(ovalo)

seed = time()  # the current timestamp
prng = Random()
prng.seed(seed)

ga = ec.GA(prng)
ga.selector = ec.selectors.tournament_selection #por defeccto
ga.variator = [ec.variators.n_point_crossover, ec.variators.random_reset_mutation] #variators para problema discreto
ga.replacer = ec.replacers.generational_replacement #por defecto
ga.terminator = ec.terminators.generation_termination
ga.observer = ec.observers.stats_observer
final_pop = ga.evolve(generator = problem.generator,
                          evaluator=problem.evaluator,
                          bounder=problem.bounder,
                          maximize=problem.maximize,
                          pop_size=100,
                          max_generations=50,
                          num_elites=1,
                          num_selected=100,
                          tournament_size=3,
                          crossover_rate=1,
                          sbx_distribution_index=10,
                          mutation_rate=0.05,
                          gaussian_stdev=0.5)

best = min(ga.population)
print('Best Solution: {0}: {1}'.format(str(best.candidate), best.fitness))