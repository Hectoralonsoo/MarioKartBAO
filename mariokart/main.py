import collections
import csv
from copy import deepcopy

import numpy as np
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
        self.miniturbo = chasis.miniturbo + ruedas.miniturbo + parapente.miniturbo + personaje.miniturbo
        self.velTierra = chasis.velTierra + ruedas.velTierra + parapente.velTierra + personaje.velTierra
        self.velAgua = chasis.velAgua + ruedas.velAgua + parapente.velAgua + personaje.velAgua
        self.velAntiGravedad = chasis.velAntiGravedad + ruedas.velAntiGravedad + parapente.velAntiGravedad + personaje.velAntiGravedad
        self.velAire = chasis.velAire + ruedas.velAire + parapente.velAire + personaje.velAire

    def printCoche(self):
        print("Chasis: " + self.chasis.nombre + " Ruedas: " + self.ruedas.nombre + " Parapente: " + self.parapente.nombre + " Personaje: " + self.personaje.nombre)

    def printStats(self):
        print("Peso: " + str(self.peso) + "\nAceleración: " + str(self.aceleracion) + "\nTracción: " + str(self.traccion) + "\nMiniturbo: " + str(self.miniturbo) + "\nVelocidad Tierra: " + str(self.velTierra) +
              "\nVelocidad Aire: " + str(self.velAire) + "\nVelocidad Agua: " + str(self.velAgua) + "\nVelocidad Antigravedad: " + str(self.velAntiGravedad))

    '''
    def calcularTiempoVuelta(self, circuito):
        tiempo = 0
        vActual = 0
        i = 0
        for tramo in circuito:
            if tramo.tipo == "recta":
                if tramo.terreno == "asfalto":
                    tiempo += calcularTiempoRecta(int(vActual), self.velTierra, calcularVMax(circuito[i+1], self), tramo.longitud,
                                                  self.aceleracion, self.peso, self.velTierra, self.traccion)
                elif tramo.terreno == "agua":
                    tiempo += calcularTiempoRecta(int(vActual), self.velAgua, calcularVMax(circuito[i+1], self), tramo.longitud,
                                                  self.aceleracion, self.peso, self.velTierra, self.traccion)
                elif tramo.terreno == "aire":
                    tiempo += calcularTiempoRecta(int(vActual), self.velAire, calcularVMax(circuito[i+1], self), tramo.longitud,
                                                  self.aceleracion, self.peso, self.velTierra, self.traccion)
                else:
                    tiempo += calcularTiempoRecta(int(vActual), self.velAntiGravedad, calcularVMax(circuito[i+1], self),
                                                  tramo.longitud, self.aceleracion, self.peso, self.velTierra,
                                                  self.traccion)

            else:
                tiempo += calcularTiempoCurva(tramo.longitud, calcularVMax(tramo, self), self.peso, self.traccion, self.velTierra)
                vActual = calcularVMax(circuito[i+1], self) + self.miniturbo * 0.9
                # parametrizar impacto del miniturbo

        return tiempo
        '''
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






def calcularTiempoVuelta(coche, circuito):
    velocidadActual = 0
    tiempo = 0
    for tramo in circuito:
        if tramo != circuito[-1]:
            if tramo == "recta":
                tiempo += calcularTiempoRecta(tramo, velocidadActual, calcularVelocidadFinalRecta(coche, tramo), coche.aceleracion)
                velocidadActual = calcularVelocidadFinalRecta(coche, tramo)
            else:
                tiempo += calcularTiempoCurva(tramo, coche)
                velocidadActual = calcularVelocidadFinalCurva(coche, tramo)
        else:
            if tramo == "recta":
                tiempo += calcularTiempoRecta(tramo, velocidadActual, calcularVelocidadFinalRecta(coche, tramo), coche.aceleracion)
            else:
                tiempo += calcularTiempoCurva(tramo, coche)
    return tiempo


def calcularTiempoAcelerando(velocidadInicial, velocidadFinal, aceleracion):
    velocidadMedia = (velocidadInicial + velocidadFinal)/2
    return calcularDistanciaAcelerando(velocidadInicial, velocidadFinal, aceleracion) /velocidadMedia

def calcularTiempoFrenando(velocidadInicial, velocidadFinal, aceleracion):
    velocidadMedia = (velocidadInicial + velocidadFinal)/2
    return calcularDistanciaAcelerando(velocidadInicial, velocidadFinal, aceleracion) /velocidadMedia



def calcularDistanciaAcelerando(velocidadInicial, velocidadFinal, aceleracion):
    return ((velocidadFinal**2 - velocidadInicial**2) / (2*aceleracion))

def calcularDistanciaFrenando(velocidadInicial, velocidadFinal, aceleracion):
    return((velocidadInicial**2-velocidadFinal**2) / (2*aceleracion))

def calcularVelocidadMaximaRecta(tramo, coche):
    if tramo.terreno == "asfalto":
        velocidad = coche.velTierra
    elif tramo.terreno == "agua":
        velocidad = coche.velAgua
    elif tramo.terreno == "aire":
        velocidad = coche.velAire
    else:
        velocidad = coche.velAntiGravedad
    return velocidad
def calcularVelocidadFinalRecta(coche, tramo):
    return calcularVelocidadCurva(tramo, coche)
def calcularVelocidadFinalCurva(coche, tramo):
    return coche.miniturbo + calcularVelocidadCurva(tramo, coche)
def tiempoDistanciaVelocidadConstante(distancia, velocidad):
    return distancia/velocidad
def calcularTiempoRecta(tramo, velocidadInicial, velocidadFinal, aceleracion):
    return calcularTiempoAcelerando(velocidadInicial, velocidadFinal, aceleracion) + tiempoDistanciaVelocidadConstante(calcularMetrosVelocidadMaximaRecta(tramo)) + calcularTiempoFrenando(velocidadInicial,velocidadFinal,aceleracion)
def calcularMetrosVelocidadMaximaRecta(tramo, velocidadInicial, velocidadFinal, aceleracion):
    if tramo.longitud <= ((calcularDistanciaAcelerando(velocidadInicial, velocidadFinal, aceleracion) + calcularDistanciaFrenando(velocidadInicial, velocidadFinal, aceleracion))):
        return 0
    else:
        return tramo.longitud - calcularDistanciaAcelerando(velocidadInicial, velocidadFinal, aceleracion) - calcularDistanciaFrenando(velocidadInicial, velocidadFinal, aceleracion)


def calcularVelocidadCurva(tramo, coche):
    if tramo.terreno == "asfalto":
        if tramo.tipo == "curva cerrada":
            velocidad = coche.velTierra - coche.peso + coche.traccion * 2 - 10
        elif tramo.tipo == "curva media":
            velocidad = coche.velTierra - coche.peso + coche.traccion * 1.5 - 7
        else:
            velocidad = coche.velTierra - coche.peso + coche.traccion * 1.25 - 5

    elif tramo.terreno == "agua":
        if tramo.tipo == "curva cerrada":
            velocidad = coche.velAgua - coche.peso + coche.traccion/2 - 10
        elif tramo.tipo == "curva media":
            velocidad = coche.velAgua - coche.peso + coche.traccion * 1.5 - 7
        else:
            velocidad = coche.velAgua - coche.peso + coche.traccion * 1.25 - 5
    elif tramo.terreno == "aire":
        if tramo.tipo == "curva cerrada":
            velocidad = coche.velAire - coche.peso + coche.traccion/2 - 10
        elif tramo.tipo == "curva media":
            velocidad = coche.velAire - coche.peso + coche.traccion * 1.5 - 7
        else:
            velocidad = coche.velAire - coche.peso + coche.traccion * 1.25 - 5
    else:
        if tramo.tipo == "curva cerrada":
            velocidad = coche.velAntiGravedad - coche.peso + coche.traccion/2 - 10
        elif tramo.tipo == "curva media":
            velocidad = coche.velAntiGravedad - coche.peso + coche.traccion * 1.5 - 7
        else:
            velocidad = coche.velAntiGravedad - coche.peso + coche.traccion * 1.25 - 5

    if (velocidad <= 0):
        return 1
    else:
        return velocidad
def calcularTiempoCurva(tramo, coche):
    return (tramo.longitud / calcularVelocidadCurva(tramo, coche))


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

#for glider in gliders:
    #print(f"Nombre: {glider.nombre}, Peso: {glider.peso}, Aceleracion: {glider.aceleracion}, Traccion: {glider.traccion}, Miniturbo: {glider.miniturbo}, Velocidad en asfalto: {glider.velTierra}, Velocidad  en aire:  {glider.velAire}, Velocidad  en antigravedad:  {glider.velAntiGravedad}, Velocidad en agua: {glider.velAgua}")

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

#for tire in tires:
    #print(f"Nombre: {tire.nombre}, Peso: {tire.peso}, Aceleracion: {tire.aceleracion}, Traccion: {tire.traccion}, Miniturbo: {tire.miniturbo}, Velocidad en asfalto: {tire.velTierra}, Velocidad en agua: {tire.velAgua}, Velocidad en antigravedad: {tire.velAntiGravedad}, Velocidad en aire: {tire.velAire}")

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

#for driver in drivers:
    #print(f"Nombre: {driver.nombre}, Peso: {driver.peso}, Aceleracion: {driver.aceleracion}, Traccion: {driver.traccion}, Miniturbo: {driver.miniturbo}, Velocidad en asfalto: {driver.velTierra}, Velocidad en agua: {driver.velAgua}, Velocidad en antigravedad: {driver.velAntiGravedad}, Velocidad en aire: {driver.velAire}")

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

#for bodie in bodies:
    #print(f"Nombre: {bodie.nombre}, Peso: {bodie.peso}, Aceleracion: {bodie.aceleracion}, Traccion: {bodie.traccion}, Miniturbo: {bodie.miniturbo}, Velocidad en asfalto: {bodie.velTierra}, Velocidad en agua: {bodie.velAgua}, Velocidad en antigravedad: {bodie.velAntiGravedad}, Velocidad en aire: {bodie.velAire}")

def generarCoche(random):
    glider = random.randint(0, 13)
    tire = random.randint(0, 20)
    driver = random.randint(0, 42)
    body = random.randint(0, 39)

    return [body, tire, glider, driver]
def ArrayToCoche(car):
    print(car)
    return Coche(bodies[car[0]],tires[car[1]],gliders[car[2]],drivers[car[3]])
def generarPoblacionInicial(size):
    poblacionInicial=[]
    for i in range(size):
        poblacionInicial.append(generarCoche())
    return poblacionInicial

'''
recta1 = Tramo(100, "agua", "recta")
curva1 = Tramo(150, "agua", "recta")
recta2 = Tramo(100, "agua", "recta")
curva2 = Tramo(150, "agua", "recta")
circuito1 = [recta1, curva1, recta2, curva2]
'''
recta1 = Tramo(100, "antigravedad", "recta")
curva1 = Tramo(150, "antigravedad", "curva cerrada")
recta2 = Tramo(100, "antigravedad", "recta")
curva2 = Tramo(100, "antigravedad", "curva cerrada")
recta3 = Tramo(200, "antigravedad", "recta")
curva3 = Tramo(100, "antigravedad", "curva media")
recta4 = Tramo(300, "antigravedad", "recta")
curva4 = Tramo(200, "antigravedad", "curva abierta")
circuito1 = [recta1, curva1, recta2, curva2, recta3, curva3, recta4, curva4]

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
        print("Bounder called with candidate:", candidate)
        bounded_candidate = candidate
        for i in range(len(bounded_candidate)):
            if bounded_candidate[i] < self.lower_bound[i]:
                bounded_candidate[i] = self.lower_bound[i]
            if bounded_candidate[i] > self.upper_bound[i]:
                bounded_candidate[i] = self.upper_bound[i]


        return bounded_candidate

def boundCandidate(candidate):
    bound = [39, 20, 13, 42]
    for i in range(len(candidate)):
        if candidate[i] < 0:
            candidate[i] = 0
        if candidate[i] > bound[i]:
            candidate[i] = bound[i]
    return candidate


class MarioKart(benchmarks.Benchmark):
    """Defines the Mario Kart benchmark problem.

    Public Attributes:

    - *circuito* -- el circuito para el que optimizar el coche


    """
    def __init__(self, circuito):
        benchmarks.Benchmark.__init__(self, 4)
        self.circuito = circuito
        #max_count = [self.capacity // item[0] for item in self.items]
        self.bounder = DiscreteBounderV2([0, 0, 0, 0], [39, 20, 13, 42])
        self.maximize = False

    def generator(self, random, args):
        """Return a candidate solution for an evolutionary algorithm."""
        return generarCoche(random)

    def evaluator(self, candidates, args):
        fitness = []
        for candidate in candidates:
            candidate = boundCandidate(candidate)
            tiempo = calcularTiempoVuelta(ArrayToCoche(candidate), self.circuito)
            fitness.append(tiempo)
        return fitness


def my_variator(random, candidates, args):
    fitness = []
    for candidate in candidates:
        candidate = boundCandidate(candidate)
        tiempo = calcularTiempoVuelta(ArrayToCoche(candidate),circuito1)
        fitness.append(tiempo)
    return candidates

size = 50


problem = MarioKart(circuito1)

seed = time()  # the current timestamp
prng = Random()
prng.seed(seed)



ga = ec.GA(prng)
ga.selector = ec.selectors.tournament_selection #por defeccto
ga.variator = [ec.variators.n_point_crossover, ec.variators.random_reset_mutation,my_variator] #variators para problema discreto
ga.replacer = ec.replacers.generational_replacement #por defecto
ga.terminator = ec.terminators.generation_termination
ga.observer = ec.observers.stats_observer
final_pop = ga.evolve(generator = problem.generator,
                          evaluator=problem.evaluator,
                          bounder=problem.bounder,
                          maximize=problem.maximize,
                          pop_size=10,
                          max_generations=20,
                          num_elites=1,
                          num_selected=20,
                          tournament_size=3,
                          crossover_rate=1,
                          sbx_distribution_index=10,
                          mutation_rate=0.05,
                          gaussian_stdev=0.5)

best = max(ga.population)
print('Best Solution: {0}: {1}'.format(str(best.candidate), best.fitness))
mejorCoche = ArrayToCoche(best.candidate)
mejorCoche.printCoche()
mejorCoche.printStats()

class ACOMarioKart:

    def __init__(self, circuito = circuito1, n_ants: int = 10, alpha: float = 1, beta: float = 5, rho: float = 0.8):

        self.circuito = circuito
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.pheromone = None
        self.best_solution = None
        self.best_fitness = None

        self.pheromone_history = []
        self.trails_history = []
        self.best_fitness_history = []

    def optimize(self, max_evaluations: int = 1000):
        self._initialize()

        n_evaluations = 0
        iter_fitness = 1e-10
        while n_evaluations < max_evaluations:
            trails = []
            for _ in range(self.n_ants):
                solution = self._construct_solution()
                fitness = self._evaluate(solution)
                n_evaluations += 1
                trails.append((solution, fitness))

                if fitness > self.best_fitness:
                    self.best_solution = solution
                    self.best_fitness = fitness

            self._update_pheromone(trails, iter_fitness)
            iter_fitness = self.best_fitness

            self.trails_history.append(deepcopy(trails))
            self.best_fitness_history.append(self.best_fitness)

        return self.best_solution

    def _initialize(self):
        self.pheromone = []
        self.pheromone.append(np.ones(len(bodies)))
        self.pheromone.append(np.ones(len(tires)))
        self.pheromone.append(np.ones(len(gliders)))
        self.pheromone.append(np.ones(len(drivers)))
        self.best_solution = None
        self.best_fitness = float('inf')

        self.pheromone_history = []
        self.trails_history = []
        self.best_fitness_history = []

    def _evaluate(self, solution: list[int]) -> float:
        solucion = ArrayToCoche(solution)
        return calcularTiempoVuelta(solucion, self.circuito)

    def _construct_solution(self) -> list[int]:
        solution = [-1, -1, -1, -1]
        i=0
        while i<4:
            while True:
                candidates = self._get_candidates(i)

                if len(candidates) == 0:
                    break
                elif len(candidates) == 1:
                    solution[i] = candidates[0]
                    break
                p = []
                p.append(self.pheromone[i])
                pheromones = p[candidates]**self.alpha
                heuristic = self._heuristic(candidates)**self.beta

                total = np.sum(pheromones * heuristic)
                probabilities = (pheromones * heuristic) / total


                solution[i] = np.random.choice(candidates, p=probabilities)
                i=i+1


        return solution

    def _heuristic(self, candidates: list[int]) -> np.ndarray:
        return np.ones(len(candidates))

    def _get_candidates(self, nPieza : int) -> np.ndarray:


        candidates = []
        if nPieza == 0:
            candidates = [i for i in range(len(bodies))]
        elif nPieza == 1:
            candidates = [i for i in range(len(tires))]
        elif nPieza == 2:
            candidates = [i for i in range(len(gliders))]
        else: candidates = [i for i in range(len(drivers))]
        return np.array(candidates)

    def _update_pheromone(self, trails: list[list[int]], best_fitness):
        self.pheromone_history.append(self.pheromone.copy())

        evaporation = 1 - self.rho
        self.pheromone *= evaporation
        for solution, fitness in trails:
            delta_fitness = 1.0/(1.0 + (best_fitness - fitness) / best_fitness)
            mask = np.argwhere(solution == 1).flatten()
            self.pheromone[mask] += delta_fitness

aco = ACOMarioKart(circuito1)
best_solution = aco.optimize()
best_coche = ArrayToCoche(best_solution)
best_coche.printCoche()
best_coche.printStats()